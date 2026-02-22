-- babelfish_redo/babelfish_engine_translategemma/init.lua
-- Translation via TranslateGemma (Gemma 3 based open translation model)
-- Based on babelfish_engine_lingva by Tai "DuCake" Kedzierski and 1F616EMO
-- Adapted for TranslateGemma with z.ai by Evgeniy aka Bad
-- SPDX-License-Identifier: LGPL-3.0-or-later

local http = assert(core.request_http_api(),
    "Could not get HTTP API table. Add babelfish_engine_translategemma to secure.http_mods")

local S = core.get_translator("babelfish_engine_translategemma")

local engine_status = "init"
local language_codes = {}
local language_alias = {}

-- Store Ollama info for logging
local ollama_info = {
    version = nil,
    model_info = nil,
    gpu_enabled = nil,
}

-- Configuration
local api_type = core.settings:get("babelfish_engine_translategemma.api_type") or "ollama"
local serviceurl = core.settings:get("babelfish_engine_translategemma.serviceurl")
local model_name = core.settings:get("babelfish_engine_translategemma.model") or "translategemma"

-- Debug logging settings
local debug_logging = core.settings:get_bool("babelfish_engine_translategemma.debug", false)
-- verbose_debug enables full model_info dump including layer details (very large output)
local verbose_debug = core.settings:get_bool("babelfish_engine_translategemma.verbose_debug", false)
-- show_license enables model license output (can be very long)
local show_license = core.settings:get_bool("babelfish_engine_translategemma.show_license", false)

-- Default URLs based on API type
if not serviceurl then
    if api_type == "ollama" then
        serviceurl = "http://localhost:11434/api/generate"
    elseif api_type == "vllm" then
        serviceurl = "http://localhost:8000/v1/completions"
    elseif api_type == "hf" then
        serviceurl = "https://api-inference.huggingface.co/models/google/translategemma-4b"
    elseif api_type == "openai" then
        serviceurl = "http://localhost:8000/v1/chat/completions"
    else
        serviceurl = "http://localhost:11434/api/generate"
    end
    core.log("warning",
        "[babelfish_engine_translategemma] babelfish_engine_translategemma.serviceurl not specified, " ..
        "using default for " .. api_type .. ": " .. serviceurl)
end

-- ============================================================================
-- Ollama Info Functions
-- ============================================================================

-- Get base URL from service URL
local function get_ollama_base_url()
    if api_type ~= "ollama" then
        return nil
    end
    -- Extract base URL (e.g., http://localhost:11434 from http://localhost:11434/api/generate)
    local base = serviceurl:match("^(https?://[^/]+)")
    return base or "http://localhost:11434"
end

-- Format bytes to human readable
local function format_bytes(bytes)
    if not bytes then return "N/A" end
    if bytes >= 1e12 then
        return string.format("%.2f TB", bytes / 1e12)
    elseif bytes >= 1e9 then
        return string.format("%.2f GB", bytes / 1e9)
    elseif bytes >= 1e6 then
        return string.format("%.2f MB", bytes / 1e6)
    elseif bytes >= 1e3 then
        return string.format("%.2f KB", bytes / 1e3)
    else
        return string.format("%d B", bytes)
    end
end

-- Create a summary of model_info without the large tensor details
local function get_model_info_summary(model_info)
    if not model_info then return "N/A" end

    local summary = {}

    -- General info
    if model_info.general then
        local general = {}
        if model_info.general.architecture then
            table.insert(general, "architecture=" .. tostring(model_info.general.architecture))
        end
        if model_info.general.file_type then
            table.insert(general, "file_type=" .. tostring(model_info.general.file_type))
        end
        if model_info.general.parameter_count then
            local params = model_info.general.parameter_count
            if params >= 1e9 then
                table.insert(general, string.format("parameters=%.1fB", params / 1e9))
            else
                table.insert(general, string.format("parameters=%.1fM", params / 1e6))
            end
        end
        if #general > 0 then
            table.insert(summary, "general: {" .. table.concat(general, ", ") .. "}")
        end
    end

    -- Llama-specific info (without tensor data)
    local llama_keys = {
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.gpu_layers",
        "llama.head_count",
        "llama.head_count_kv",
        "llama.layer_count",
        "llama.rope.dimension_count",
    }

    local llama_info = {}
    for _, key in ipairs(llama_keys) do
        if model_info[key] ~= nil then
            local short_key = key:gsub("llama%.", "")
            table.insert(llama_info, short_key .. "=" .. tostring(model_info[key]))
        end
    end
    if #llama_info > 0 then
        table.insert(summary, "llama: {" .. table.concat(llama_info, ", ") .. "}")
    end

    return "{\n    " .. table.concat(summary, ",\n    ") .. "\n  }"
end

-- Log Ollama version info
local function log_ollama_version()
    local base_url = get_ollama_base_url()
    if not base_url then return end

    http.fetch({
        url = base_url .. "/api/version",
        method = "GET",
        timeout = 5,
    }, function(response)
        local success, err = pcall(function()
            if response.succeeded then
                local data = core.parse_json(response.data)
                if data then
                    ollama_info.version = data
                    core.log("action", "[babelfish_engine_translategemma] Ollama version: " .. (data.version or "unknown"))
                    if debug_logging then
                        core.log("action", "[babelfish_engine_translategemma] Full version info: " .. dump(data))
                    end
                end
            else
                core.log("warning", "[babelfish_engine_translategemma] Could not fetch Ollama version")
            end
        end)
        if not success then
            core.log("warning", "[babelfish_engine_translategemma] Error processing version info: " .. tostring(err))
        end
    end)
end

-- Log Ollama model info (GPU, parameters, etc.)
local function log_ollama_model_info()
    local base_url = get_ollama_base_url()
    if not base_url then return end

    http.fetch({
        url = base_url .. "/api/show",
        method = "POST",
        timeout = 10,
        extra_headers = { "Content-Type: application/json" },
        post_data = core.write_json({ name = model_name }),
    }, function(response)
        -- Wrap in pcall to prevent crashes from unexpected data structures
        local success, err = pcall(function()
            if not response.succeeded then
                core.log("warning", "[babelfish_engine_translategemma] Could not fetch model info for: " .. model_name)
                core.log("warning", "[babelfish_engine_translategemma] Make sure the model is pulled: ollama pull " .. model_name)
                return
            end

            local data = core.parse_json(response.data)
            if not data then
                core.log("warning", "[babelfish_engine_translategemma] Could not parse model info response")
                return
            end

            ollama_info.model_info = data

            -- Log model details
            core.log("action", "[babelfish_engine_translategemma] Model: " .. model_name)

            -- License info (can be very long, disabled by default)
            if data.license and show_license then
                core.log("action", "[babelfish_engine_translategemma] Model license: " .. tostring(data.license))
            elseif data.license then
                core.log("action", "[babelfish_engine_translategemma] Model license: (available, enable show_license=true to display)")
            end

            -- Modelfile parameters
            if data.parameters then
                core.log("action", "[babelfish_engine_translategemma] Model parameters: " .. tostring(data.parameters))
            end

            -- Template
            if data.template and debug_logging then
                core.log("action", "[babelfish_engine_translategemma] Model template: " .. tostring(data.template))
            end

            -- Model details (size, format, family, etc.)
            if data.details then
                local details = data.details
                core.log("action", "[babelfish_engine_translategemma] Model format: " .. (details.format or "N/A"))
                core.log("action", "[babelfish_engine_translategemma] Model family: " .. (details.family or "N/A"))
                core.log("action", "[babelfish_engine_translategemma] Parameter count: " .. (details.parameter_size or "N/A"))
                core.log("action", "[babelfish_engine_translategemma] Quantization: " .. (details.quantization_level or "N/A"))
            end

            -- Model info (contains GPU and memory info)
            if data.model_info then
                local info = data.model_info

                -- General info (check if exists first)
                if info.general then
                    if info.general.architecture then
                        core.log("action", "[babelfish_engine_translategemma] Architecture: " .. tostring(info.general.architecture))
                    end
                    if info.general.file_type then
                        core.log("action", "[babelfish_engine_translategemma] File type: " .. tostring(info.general.file_type))
                    end

                    -- Parameter count
                    if info.general.parameter_count then
                        local params = info.general.parameter_count
                        if params >= 1e9 then
                            core.log("action", "[babelfish_engine_translategemma] Parameters: " ..
                                string.format("%.1fB", params / 1e9))
                        else
                            core.log("action", "[babelfish_engine_translategemma] Parameters: " ..
                                string.format("%.1fM", params / 1e6))
                        end
                    end
                end

                -- Context length
                if info["llama.context_length"] or info["llama.block_count"] then
                    local ctx_len = info["llama.context_length"] or info["llama.block_count"]
                    core.log("action", "[babelfish_engine_translategemma] Context length: " .. tostring(ctx_len))
                end

                -- Embedding length
                if info["llama.embedding_length"] then
                    core.log("action", "[babelfish_engine_translategemma] Embedding length: " ..
                        tostring(info["llama.embedding_length"]))
                end

                -- Check GPU layers
                if info["llama.gpu_layers"] then
                    local gpu_layers = info["llama.gpu_layers"]
                    ollama_info.gpu_enabled = gpu_layers > 0
                    if gpu_layers > 0 then
                        core.log("action", "[babelfish_engine_translategemma] GPU acceleration: ENABLED (" ..
                            gpu_layers .. " layers on GPU)")
                    else
                        core.log("warning", "[babelfish_engine_translategemma] GPU acceleration: DISABLED (0 layers on GPU)")
                    end
                end
            end

            -- Debug: log model info summary (use verbose_debug for full dump with layer details)
            if verbose_debug then
                core.log("action", "[babelfish_engine_translategemma] Full model info (verbose): " .. dump(data))
            elseif debug_logging then
                -- Log summary without huge tensor data
                if data.model_info then
                    core.log("action", "[babelfish_engine_translategemma] Model info summary: " ..
                        get_model_info_summary(data.model_info))
                end
            end
        end)

        if not success then
            core.log("warning", "[babelfish_engine_translategemma] Error processing model info: " .. tostring(err))
            if debug_logging then
                core.log("action", "[babelfish_engine_translategemma] Raw response: " .. (response.data or "nil"))
            end
        end
    end)
end

-- Log running models (GPU memory usage, etc.)
local function log_ollama_running_models()
    local base_url = get_ollama_base_url()
    if not base_url then return end

    http.fetch({
        url = base_url .. "/api/ps",
        method = "GET",
        timeout = 5,
    }, function(response)
        if response.succeeded then
            local data = core.parse_json(response.data)
            if data and data.models and #data.models > 0 then
                core.log("action", "[babelfish_engine_translategemma] Currently loaded models:")
                for _, model in ipairs(data.models) do
                    local size_vram = format_bytes(model.size_vram)
                    local size_ram = format_bytes(model.size)
                    local until_exp = model.expires_at or "N/A"

                    if model.size_vram and model.size_vram > 0 then
                        core.log("action", string.format(
                            "[babelfish_engine_translategemma]   - %s: VRAM=%s, RAM=%s, expires=%s",
                            model.name, size_vram, size_ram, until_exp
                        ))
                    else
                        core.log("action", string.format(
                            "[babelfish_engine_translategemma]   - %s: RAM=%s, expires=%s (no VRAM usage)",
                            model.name, size_ram, until_exp
                        ))
                    end
                end
            else
                core.log("action", "[babelfish_engine_translategemma] No models currently loaded in memory")
            end
        else
            core.log("warning", "[babelfish_engine_translategemma] Could not fetch running models")
        end
    end)
end

-- Log all Ollama info
local function log_ollama_startup_info()
    if api_type ~= "ollama" then
        core.log("action", "[babelfish_engine_translategemma] API type: " .. api_type .. " (Ollama info not applicable)")
        return
    end

    core.log("action", "[babelfish_engine_translategemma] ========== Ollama Startup Info ==========")
    core.log("action", "[babelfish_engine_translategemma] Ollama API URL: " .. get_ollama_base_url())
    core.log("action", "[babelfish_engine_translategemma] Model name: " .. model_name)

    -- Fetch and log all info
    log_ollama_version()
    log_ollama_model_info()
    log_ollama_running_models()

    core.log("action", "[babelfish_engine_translategemma] =======================================")
end

-- TranslateGemma supports 55 languages
-- Language codes based on TranslateGemma documentation
local supported_languages = {
    "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es",
    "et", "fa", "fi", "fr", "he", "hi", "hr", "hu", "id", "it",
    "ja", "ko", "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro",
    "ru", "sk", "sl", "sr", "sv", "th", "tl", "tr", "uk", "vi",
    "zh", "af", "am", "az", "be", "bs", "cy", "eo", "eu", "gl",
    "gu", "hy", "ka", "kn", "ml", "mr", "ne", "pa", "si", "sq",
    "sw", "ta", "te", "ur", "uz"
}

-- Language names for display
local language_names = {
    ["ar"] = "Arabic", ["bg"] = "Bulgarian", ["bn"] = "Bengali", ["ca"] = "Catalan",
    ["cs"] = "Czech", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek",
    ["en"] = "English", ["es"] = "Spanish", ["et"] = "Estonian", ["fa"] = "Persian",
    ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew", ["hi"] = "Hindi",
    ["hr"] = "Croatian", ["hu"] = "Hungarian", ["id"] = "Indonesian", ["it"] = "Italian",
    ["ja"] = "Japanese", ["ko"] = "Korean", ["lt"] = "Lithuanian", ["lv"] = "Latvian",
    ["ms"] = "Malay", ["nl"] = "Dutch", ["no"] = "Norwegian", ["pl"] = "Polish",
    ["pt"] = "Portuguese", ["ro"] = "Romanian", ["ru"] = "Russian", ["sk"] = "Slovak",
    ["sl"] = "Slovenian", ["sr"] = "Serbian", ["sv"] = "Swedish", ["th"] = "Thai",
    ["tl"] = "Tagalog", ["tr"] = "Turkish", ["uk"] = "Ukrainian", ["vi"] = "Vietnamese",
    ["zh"] = "Chinese", ["af"] = "Afrikaans", ["am"] = "Amharic", ["az"] = "Azerbaijani",
    ["be"] = "Belarusian", ["bs"] = "Bosnian", ["cy"] = "Welsh", ["eo"] = "Esperanto",
    ["eu"] = "Basque", ["gl"] = "Galician", ["gu"] = "Gujarati", ["hy"] = "Armenian",
    ["ka"] = "Georgian", ["kn"] = "Kannada", ["ml"] = "Malayalam", ["mr"] = "Marathi",
    ["ne"] = "Nepali", ["pa"] = "Punjabi", ["si"] = "Sinhala", ["sq"] = "Albanian",
    ["sw"] = "Swahili", ["ta"] = "Tamil", ["te"] = "Telugu", ["ur"] = "Urdu",
    ["uz"] = "Uzbek"
}

-- Build language codes and aliases
do
    local valid_alias = {
        ["zh"] = {
            "zhs",
            "zh-cn",
            "zh-hans",
            "chinese",
        },
        ["en"] = {
            "english",
        },
        ["ru"] = {
            "russian",
        },
        ["de"] = {
            "german",
        },
        ["fr"] = {
            "french",
        },
        ["es"] = {
            "spanish",
        },
        ["ja"] = {
            "japanese",
        },
        ["ko"] = {
            "korean",
        },
    }

    local alias_log_strings = {}

    for _, code in ipairs(supported_languages) do
        language_codes[code] = language_names[code] or code

        if valid_alias[code] then
            for _, alias in ipairs(valid_alias[code]) do
                language_alias[alias] = code
                alias_log_strings[#alias_log_strings + 1] = alias .. " -> " .. code
            end
        end
    end

    core.log("action", "[babelfish_engine_translategemma] Supported languages: " ..
        #supported_languages .. " languages")
    core.log("action", "[babelfish_engine_translategemma] Got language alias: " ..
        table.concat(alias_log_strings, "; "))

    engine_status = "ready"

    -- Log Ollama startup info after initialization
    log_ollama_startup_info()
end

-- Build translation prompt for TranslateGemma
local function build_translation_prompt(source, target, query)
    local source_lang = language_names[source] or source
    local target_lang = language_names[target] or target

    if source == "auto" then
        -- Auto-detect language
        return string.format(
            "Translate the following text to %s. Provide only the translation, no explanations:\n\n%s",
            target_lang, query
        )
    else
        return string.format(
            "Translate the following text from %s to %s. Provide only the translation, no explanations:\n\n%s",
            source_lang, target_lang, query
        )
    end
end

-- Parse response based on API type
local function parse_response(api_type, data)
    if api_type == "ollama" then
        return data.response
    elseif api_type == "vllm" or api_type == "openai" then
        if data.choices and data.choices[1] then
            if data.choices[1].text then
                return data.choices[1].text
            elseif data.choices[1].message then
                return data.choices[1].message.content
            end
        end
    elseif api_type == "hf" then
        if type(data) == "table" and data[1] and data[1].generated_text then
            return data[1].generated_text
        elseif type(data) == "string" then
            return data
        end
    end
    return nil
end

-- Make API request based on type
local function make_request(source, target, query, callback)
    local prompt = build_translation_prompt(source, target, query)
    local post_data
    local extra_headers = {}
    local request_start = os.time()

    -- Log translation request
    core.log("action", string.format("[babelfish_engine_translategemma] Translation request: %s -> %s, text length: %d chars",
        source, target, #query))

    if api_type == "ollama" then
        -- Get optional generation parameters from settings
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local num_predict = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024
        local num_ctx = tonumber(core.settings:get("babelfish_engine_translategemma.num_ctx")) or 2048
        local top_k = tonumber(core.settings:get("babelfish_engine_translategemma.top_k")) or 40
        local top_p = tonumber(core.settings:get("babelfish_engine_translategemma.top_p")) or 0.9
        local num_gpu = tonumber(core.settings:get("babelfish_engine_translategemma.num_gpu")) or -1  -- -1 = auto

        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            stream = false,
            options = {
                temperature = temperature,
                num_predict = num_predict,
                num_ctx = num_ctx,
                top_k = top_k,
                top_p = top_p,
                num_gpu = num_gpu,
            }
        })

        if debug_logging then
            core.log("action", "[babelfish_engine_translategemma] Generation params: " ..
                string.format("temp=%.2f, num_predict=%d, num_ctx=%d, top_k=%d, top_p=%.2f, num_gpu=%d",
                    temperature, num_predict, num_ctx, top_k, top_p, num_gpu))
        end
    elseif api_type == "vllm" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024

        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
        })
        extra_headers = { "Content-Type: application/json" }

        if debug_logging then
            core.log("action", "[babelfish_engine_translategemma] Generation params: " ..
                string.format("temp=%.2f, max_tokens=%d", temperature, max_tokens))
        end
    elseif api_type == "openai" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024

        post_data = core.write_json({
            model = model_name,
            messages = {
                { role = "user", content = prompt }
            },
            max_tokens = max_tokens,
            temperature = temperature,
        })
        extra_headers = { "Content-Type: application/json" }

        if debug_logging then
            core.log("action", "[babelfish_engine_translategemma] Generation params: " ..
                string.format("temp=%.2f, max_tokens=%d", temperature, max_tokens))
        end
    elseif api_type == "hf" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024

        post_data = core.write_json({
            inputs = prompt,
            parameters = {
                max_new_tokens = max_tokens,
                temperature = temperature,
                return_full_text = false,
            }
        })
        extra_headers = { "Content-Type: application/json" }

        -- Check for HF token in settings
        local hf_token = core.settings:get("babelfish_engine_translategemma.hf_token")
        if hf_token then
            table.insert(extra_headers, "Authorization: Bearer " .. hf_token)
        end

        if debug_logging then
            core.log("action", "[babelfish_engine_translategemma] Generation params: " ..
                string.format("temp=%.2f, max_new_tokens=%d", temperature, max_tokens))
        end
    else
        -- Default to Ollama format
        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            stream = false,
        })
    end

    return http.fetch({
        url = serviceurl,
        method = "POST",
        timeout = 30,
        extra_headers = extra_headers,
        post_data = post_data,
    }, function(response)
        local request_time = os.difftime(os.time(), request_start)

        if not response.succeeded then
            core.log("error", "[babelfish_engine_translategemma] HTTP request failed after " .. request_time .. "s: " .. dump(response))
            return callback(false)
        end

        local data, err = core.parse_json(response.data, nil, true)
        if not data then
            core.log("error", "[babelfish_engine_translategemma] JSON parse error: " .. err)
            core.log("error", "[babelfish_engine_translategemma] Raw data: " .. response.data)
            return callback(false)
        end

        local translation = parse_response(api_type, data)

        -- Log performance metrics
        if api_type == "ollama" and data then
            local total_duration = data.total_duration and (data.total_duration / 1e9) or 0
            local load_duration = data.load_duration and (data.load_duration / 1e9) or 0
            local prompt_eval_count = data.prompt_eval_count or 0
            local eval_count = data.eval_count or 0
            local eval_duration = data.eval_duration and (data.eval_duration / 1e9) or 0

            core.log("action", string.format(
                "[babelfish_engine_translategemma] Response in %.2fs (load: %.2fs, eval: %.2fs), " ..
                "tokens: prompt=%d, generated=%d, speed: %.1f tok/s",
                total_duration, load_duration, eval_duration,
                prompt_eval_count, eval_count,
                eval_count > 0 and eval_count / eval_duration or 0
            ))

            if debug_logging and data.context then
                core.log("action", "[babelfish_engine_translategemma] Full response: " .. dump(data))
            end
        else
            core.log("action", "[babelfish_engine_translategemma] Response received in " .. request_time .. "s")
        end

        if translation then
            -- Clean up the translation (remove any leading/trailing whitespace and quotes)
            translation = translation:match("^%s*(.-)%s*$")
            translation = translation:gsub('^"', ''):gsub('"$', '')

            core.log("action", "[babelfish_engine_translategemma] Translation successful: " ..
                #translation .. " chars output")

            return callback(true, translation, source == "auto" and "auto" or source)
        end

        return callback(false, S("Error parsing translation response"))
    end)
end

---Function for translating a given text
---@param source string Source language code. If `"auto"`, detect the language automatically.
---@param target string Target language code.
---@param query string String to translate.
---@param callback BabelFishCallback Callback to run after finishing (or failing) a request
local function translate(source, target, query, callback)
    if engine_status == "error" then
        return callback(false, S("Engine error while initializing."))
    elseif engine_status == "init" then
        return callback(false, S("Engine not yet initialized."))
    end

    -- Handle Chinese variants
    if source == "zh_HANT" or source == "zh-tw" or source == "zh-hant" then
        source = "zh"  -- TranslateGemma handles Chinese as one language
    end
    if target == "zh_HANT" or target == "zh-tw" or target == "zh-hant" then
        target = "zh"
    end

    -- Validate language codes
    if source ~= "auto" and not language_codes[source] then
        -- Check aliases
        local resolved = language_alias[source:lower()]
        if resolved then
            source = resolved
        else
            core.log("warning", "[babelfish_engine_translategemma] Unknown source language: " .. source)
            -- Continue anyway, the model might still work
        end
    end

    if not language_codes[target] then
        local resolved = language_alias[target:lower()]
        if resolved then
            target = resolved
        else
            core.log("warning", "[babelfish_engine_translategemma] Unknown target language: " .. target)
        end
    end

    return make_request(source, target, query, callback)
end

-- Minetest to TranslateGemma language code mapping
-- Maps common locale codes (e.g., ru_RU, de_DE) to engine codes (e.g., ru, de)
local mt_language_map = {
    -- Existing / Specific overrides
    ["es_US"] = "es",
    ["lzh"] = "zh",
    ["zh_CN"] = "zh",
    ["zh_TW"] = "zh",
    ["sr_Cyrl"] = "sr",
    ["sr_Latn"] = "sr",

    -- Russian
    ["ru_RU"] = "ru",

    -- English variants
    ["en_US"] = "en",
    ["en_GB"] = "en",

    -- European languages
    ["de_DE"] = "de",
    ["de_AT"] = "de",
    ["de_CH"] = "de",
    ["fr_FR"] = "fr",
    ["fr_CA"] = "fr",
    ["fr_CH"] = "fr",
    ["it_IT"] = "it",
    ["it_CH"] = "it",
    ["es_ES"] = "es",
    ["es_MX"] = "es",
    ["es_AR"] = "es",
    ["pt_PT"] = "pt",
    ["pt_BR"] = "pt",
    ["nl_NL"] = "nl",
    ["nl_BE"] = "nl",
    ["pl_PL"] = "pl",
    ["uk_UA"] = "uk",
    ["cs_CZ"] = "cs",
    ["sk_SK"] = "sk",
    ["hu_HU"] = "hu",
    ["ro_RO"] = "ro",
    ["bg_BG"] = "bg",
    ["hr_HR"] = "hr",
    ["sr_RS"] = "sr", -- Serbian (Cyrillic)
    ["sl_SI"] = "sl",
    ["et_EE"] = "et",
    ["lv_LV"] = "lv",
    ["lt_LT"] = "lt",
    ["el_GR"] = "el",
    ["fi_FI"] = "fi",
    ["sv_SE"] = "sv",
    ["da_DK"] = "da",
    ["no_NO"] = "no",
    ["nb_NO"] = "no", -- Norwegian Bokmål
    ["nn_NO"] = "no", -- Norwegian Nynorsk
    ["tr_TR"] = "tr",
    ["sq_AL"] = "sq", -- Albanian
    ["bs_BA"] = "bs", -- Bosnian
    ["ca_ES"] = "ca", -- Catalan
    ["eu_ES"] = "eu", -- Basque
    ["gl_ES"] = "gl", -- Galician
    ["mt_MT"] = "mt", -- Maltese (if supported by model, else remove)
    ["cy_GB"] = "cy", -- Welsh

    -- Asian and Middle Eastern languages
    ["ja_JP"] = "ja",
    ["ko_KR"] = "ko",
    ["zh_HK"] = "zh",
    ["zh_SG"] = "zh",
    ["ar_SA"] = "ar",
    ["ar_EG"] = "ar",
    ["he_IL"] = "he",
    ["hi_IN"] = "hi",
    ["bn_BD"] = "bn", -- Bengali (Bangladesh)
    ["bn_IN"] = "bn", -- Bengali (India)
    ["fa_IR"] = "fa", -- Persian
    ["ur_PK"] = "ur", -- Urdu
    ["ta_IN"] = "ta", -- Tamil
    ["te_IN"] = "te", -- Telugu
    ["mr_IN"] = "mr", -- Marathi
    ["gu_IN"] = "gu", -- Gujarati
    ["kn_IN"] = "kn", -- Kannada
    ["ml_IN"] = "ml", -- Malayalam
    ["pa_IN"] = "pa", -- Punjabi (Gurmukhi)
    ["pa_PK"] = "pa", -- Punjabi (Shahmukhi, might map differently but usually 'pa' covers broad)
    ["th_TH"] = "th",
    ["vi_VN"] = "vi",
    ["id_ID"] = "id",
    ["ms_MY"] = "ms", -- Malay
    ["tl_PH"] = "tl", -- Tagalog
    ["sw_KE"] = "sw", -- Swahili
    ["ne_NP"] = "ne", -- Nepali
    ["si_LK"] = "si", -- Sinhala

    -- Other regions
    ["af_ZA"] = "af", -- Afrikaans
    ["am_ET"] = "am", -- Amharic
    ["hy_AM"] = "hy", -- Armenian
    ["az_AZ"] = "az", -- Azerbaijani
    ["be_BY"] = "be", -- Belarusian
    ["ka_GE"] = "ka", -- Georgian
    ["kk_KZ"] = "kk", -- Kazakh (Note: not in supported list, but good practice to map if model updates)
    ["uz_UZ"] = "uz", -- Uzbek
}

babelfish.register_engine({
    translate = translate,
    language_codes = language_codes,
    language_alias = language_alias,
    mt_language_map = mt_language_map,

    compliance = nil,
    engine_label = "TranslateGemma",
})

core.log("action", "[babelfish_engine_translategemma] Engine registered successfully with API type: " .. api_type)
