// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "onnxruntime_loader.h"

#include <codecvt>
#include <future>
#include <locale>
#include <string>
#include <thread>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <fstream>
#include <vector>
#include <cstdlib>

#include "onnxruntime_utils.h"

namespace triton { namespace backend { namespace onnxruntime {

std::unique_ptr<OnnxLoader> OnnxLoader::loader = nullptr;

OnnxLoader::~OnnxLoader()
{
  if (env_ != nullptr) {
    ort_api->ReleaseEnv(env_);
  }
}

TRITONSERVER_Error*
OnnxLoader::Init(common::TritonJson::Value& backend_config)
{
  if (loader == nullptr) {
    OrtEnv* env;
    // If needed, provide custom logger with
    // ort_api->CreateEnvWithCustomLogger()
    OrtStatus* status;
    OrtLoggingLevel logging_level =
        TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)
            ? ORT_LOGGING_LEVEL_VERBOSE
        : TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_WARN)
            ? ORT_LOGGING_LEVEL_WARNING
            : ORT_LOGGING_LEVEL_ERROR;

    // Controls whether to enable global threadpool which will be shared across
    // sessions. Use this in conjunction with DisablePerSessionThreads API or
    // else the session will use it's own thread pool.
    bool global_threadpool_enabled = false;
    OrtThreadingOptions* threading_options = nullptr;

    // Read backend config
    triton::common::TritonJson::Value cmdline;
    if (backend_config.Find("cmdline", &cmdline)) {
      triton::common::TritonJson::Value value;
      std::string value_str;
      if (cmdline.Find("enable-global-threadpool", &value)) {
        RETURN_IF_ERROR(value.AsString(&value_str));
        RETURN_IF_ERROR(ParseBoolValue(value_str, &global_threadpool_enabled));

        if (global_threadpool_enabled) {
          // If provided by user, read intra and inter op num thread
          // configuration and set ThreadingOptions accordingly. If not, we use
          // default 0 which means value equal to number of cores will be used.
          RETURN_IF_ORT_ERROR(
              ort_api->CreateThreadingOptions(&threading_options));
          if (cmdline.Find("intra_op_thread_count", &value)) {
            int intra_op_num_threads = 0;
            RETURN_IF_ERROR(value.AsString(&value_str));
            RETURN_IF_ERROR(ParseIntValue(value_str, &intra_op_num_threads));
            if (intra_op_num_threads > 0) {
              RETURN_IF_ORT_ERROR(ort_api->SetGlobalIntraOpNumThreads(
                  threading_options, intra_op_num_threads));
            }
          }
          if (cmdline.Find("inter_op_thread_count", &value)) {
            int inter_op_num_threads = 0;
            RETURN_IF_ERROR(value.AsString(&value_str));
            RETURN_IF_ERROR(ParseIntValue(value_str, &inter_op_num_threads));
            if (inter_op_num_threads > 0) {
              RETURN_IF_ORT_ERROR(ort_api->SetGlobalInterOpNumThreads(
                  threading_options, inter_op_num_threads));
            }
          }
        }
      }
    }

    if (global_threadpool_enabled && threading_options != nullptr) {
      status = ort_api->CreateEnvWithGlobalThreadPools(
          logging_level, "log", threading_options, &env);
      ort_api->ReleaseThreadingOptions(threading_options);
    } else {
      status = ort_api->CreateEnv(logging_level, "log", &env);
    }

    loader.reset(new OnnxLoader(env, global_threadpool_enabled));
    RETURN_IF_ORT_ERROR(status);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        "OnnxLoader singleton already initialized");
  }

  return nullptr;  // success
}

void
OnnxLoader::TryRelease(bool decrement_session_cnt)
{
  std::unique_ptr<OnnxLoader> lloader;
  {
    std::lock_guard<std::mutex> lk(loader->mu_);
    if (decrement_session_cnt) {
      loader->live_session_cnt_--;
    }

    if (loader->closing_ && (loader->live_session_cnt_ == 0)) {
      lloader.swap(loader);
    }
  }
}

TRITONSERVER_Error*
OnnxLoader::Stop()
{
  if (loader != nullptr) {
    loader->closing_ = true;
    TryRelease(false);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }

  return nullptr;  // success
}

bool
OnnxLoader::IsGlobalThreadPoolEnabled()
{
  if (loader != nullptr) {
    return loader->global_threadpool_enabled_;
  }

  return false;
}

std::vector<unsigned char> get_decrypt_key_and_iv_from_env() {
    const char* key_env = std::getenv("DECRYPT_KEY");
    if (!key_env) {
        return {};
    }
    std::string key_hex = std::string(key_env);
    
    if (key_hex.size() != 96) {
        throw std::runtime_error("The key length must be 96");
    }

    std::vector<unsigned char> key_bytes;
    for (size_t i = 0; i < key_hex.size(); i += 2) {
        std::string byte_string = key_hex.substr(i, 2);
        unsigned char byte = static_cast<unsigned char>(std::stoi(byte_string, nullptr, 16));
        key_bytes.push_back(byte);
    }
    return key_bytes;
}

std::vector<unsigned char> decrypt(const std::vector<unsigned char>& encrypted_data, const std::vector<unsigned char>& key_and_iv) {
    constexpr size_t key_size = 32; // in bytes or 64 in hex
    constexpr size_t iv_size = 16;

    if (key_and_iv.size() != key_size + iv_size) {
        throw std::invalid_argument("key_and_iv must contain both key and IV with lengths 32 + 16 bytes.");
    }

    const unsigned char* key = key_and_iv.data();
    const unsigned char* iv = key_and_iv.data() + key_size;

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    std::vector<unsigned char> decrypted_data(encrypted_data.size());
    EVP_DecryptInit_ex(ctx, EVP_aes_256_cfb(), NULL, key, iv);

    int outlen1 = 0;
    EVP_DecryptUpdate(ctx, decrypted_data.data(), &outlen1, 
                          encrypted_data.data(), 
                          static_cast<int>(encrypted_data.size()));

    int outlen2 = 0;
    EVP_DecryptFinal_ex(ctx, decrypted_data.data() + outlen1, &outlen2);
    EVP_CIPHER_CTX_free(ctx);

    decrypted_data.resize(outlen1 + outlen2);
    return decrypted_data;
}

std::vector<unsigned char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Can't open file: " + filename);
    }

    std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
    return buffer;
}

TRITONSERVER_Error*
OnnxLoader::LoadSession(
    bool is_path, const std::string& model,
    const OrtSessionOptions* session_options, OrtSession** session)
{
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::wstring ort_style_model_str = converter.from_bytes(model);
#else
  std::string ort_style_model_str = model;
#endif
  std::vector<unsigned char> key = get_decrypt_key_and_iv_from_env();
  if (!key.empty() && is_path) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Decrypting model");
    std::vector<unsigned char> model_bytes = read_file(ort_style_model_str);
    std::vector<unsigned char> decrypted_model = decrypt(model_bytes, key);
    ort_style_model_str = std::string(decrypted_model.begin(), decrypted_model.end());
    is_path = false;
  }
  if (loader != nullptr) {
    {
      std::lock_guard<std::mutex> lk(loader->mu_);
      if (loader->closing_) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, "OnnxLoader has been stopped");
      } else {
        loader->live_session_cnt_++;
      }
    }

    OrtStatus* status = nullptr;
    {
      // [FIXME] Remove lock when ORT create session is thread safe [DLIS-4663]
      static std::mutex ort_create_session_mu;
      std::lock_guard<std::mutex> ort_lk(ort_create_session_mu);

      if (!is_path) {
        status = ort_api->CreateSessionFromArray(
            loader->env_, ort_style_model_str.c_str(), ort_style_model_str.size(),
            session_options, session);
      } else {
        status = ort_api->CreateSession(
            loader->env_, ort_style_model_str.c_str(), session_options,
            session);
      }
    }

    if (status != nullptr) {
      TryRelease(true);
    }
    RETURN_IF_ORT_ERROR(status);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
OnnxLoader::UnloadSession(OrtSession* session)
{
  if (loader != nullptr) {
    ort_api->ReleaseSession(session);
    TryRelease(true);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "OnnxLoader singleton has not been initialized");
  }

  return nullptr;  // success
}

}}}  // namespace triton::backend::onnxruntime
