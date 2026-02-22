-- babelfish_redo/babelfish_engine_lingva/init.lua
-- Google Translate via the Lingva frontend
-- Copyright (C) 2016  Tai "DuCake" Kedzierski
-- Copyright (C) 2024  1F616EMO
-- SPDX-License-Identifier: LGPL-3.0-or-later

local http = assert(core.request_http_api(),
    "Could not get HTTP API table. Add babelfish_engine_lingva to secure.http_mods")

local S = core.get_translator("babelfish_engine_lingva")

local engine_status = "init"
local language_codes = {}
local language_alias = {}

local serviceurl = core.settings:get("babelfish_engine_lingva.serviceurl")
if not serviceurl then
    serviceurl = "https://lingva.ml/api/graphql"
    core.log("warning",
        "[babelfish_engine_lingva] babelfish_engine_lingva.serviceurl not specified, " ..
        "using official instance (https://lingva.ml/api/graphql)")
end

local function graphql_fetch(query, func)
    return http.fetch({
        url = serviceurl,
        method = "POST",
        timeout = 10,
        extra_headers = { "accept: application/graphql-response+json;charset=utf-8, application/json;charset=utf-8" },
        post_data = core.write_json({
            query = query
        }),
    }, function(responce)
        if not responce.succeeded then
            core.log("error", "[babelfish_engine_lingva] Error on requesting " .. query .. ": " .. dump(responce))
            return func(false)
        end

        local data, err = core.parse_json(responce.data, nil, true)
        if not data then
            core.log("error", "[babelfish_engine_lingva] Error on requesting " .. query .. ": " .. err)
            core.log("error", "[babelfish_engine_lingva] Raw data: " .. responce.data)
            return func(false)
        end

        if data.errors then
            core.log("error", "[babelfish_engine_lingva] Error on requesting " .. query .. ": ")
            for i, error in ipairs(data.errors) do
                local location_string = "?"
                if error.locations then
                    local location_strings = {}
                    for _, location in ipairs(error.locations) do
                        location_strings[#location_strings + 1] = location.line .. ":" .. location.column
                    end
                    location_string = table.concat(location_strings, ", ")
                end
                core.log("error", string.format("[babelfish_engine_lingva] (%d/%d) Line(s) %s: %s (%s)",
                    i, #data.errors,
                    location_string, error.message, error.extensions and error.extensions.code or "UNKNOWN"))

                if error.extensions and error.extensions.stacktrace then
                    core.log("error", "[babelfish_engine_lingva]Stacktrace:")
                    for _, line in ipairs(error.extensions.stacktrace) do
                        core.log("error", "[babelfish_engine_lingva] \t" .. line)
                    end
                end
            end
        end

        if not data.data then
            return func(false)
        end

        return func(data.data)
    end)
end

do
    local valid_alias = {
        ["zh_HANT"] = {
            "zht",
            "zh-tw",
            "zh-hant",
        },
        ["zh"] = {
            "zhs",
            "zh-cn",
            "zh-hans",
        },
    }

    graphql_fetch("{languages{code,name}}", function(data)
        if not data then
            engine_status = "error"
            return
        end

        local langs_got = {}
        local alias_log_strings = {}
        -- We assume all langauge supports bidirectional translation
        for _, langdata in ipairs(data.languages) do
            if langdata.code ~= "auto" then
                language_codes[langdata.code] = langdata.name
                langs_got[#langs_got + 1] = langdata.code

                if valid_alias[langdata.code] then
                    for _, alias in ipairs(valid_alias[langdata.code]) do
                        language_alias[alias] = langdata.code
                        alias_log_strings[#alias_log_strings + 1] =
                            alias .. " -> " .. langdata.code
                    end
                end
            end
        end
        core.log("action", "[babelfish_engine_lingva] Got language list: " .. table.concat(langs_got, ", "))
        core.log("action", "[babelfish_engine_lingva] Got language alias: " .. table.concat(alias_log_strings, "; "))
        engine_status = "ready"
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

    if source == "zh_HANT" then
        -- Lingva doesn't support specifying Chinese variant, use zh
        source = "zh"
    end

    -- query = string.gsub(query, "\"", "\\\"")
    graphql_fetch(
        "{translation(source: \"" .. source .. "\", target: \"" .. target ..
        "\", query: \"\"\"" .. query .. "\"\"\"){source{detected{code}},target{text}}}",
        function(data)
            if data then
                return callback(
                    true,
                    data.translation.target.text,
                    data.translation.source.detected and data.translation.source.detected.code or source
                )
            end
            return callback(false, S("Error getting translation"))
        end)
end

local mt_language_map = {
    -- Existing specific overrides
    ["es_US"] = "es",
    ["lzh"] = "zh_HANT",
    ["zh_CN"] = "zh",
    ["zh_TW"] = "zh_HANT",
    ["sr_Cyrl"] = "sr",
    ["sr_Latn"] = "sr",

    -- Russian
    ["ru_RU"] = "ru",

    -- English variants
    ["en_US"] = "en",
    ["en_GB"] = "en",
    ["en_AU"] = "en",
    ["en_CA"] = "en",

    -- European languages
    ["de_DE"] = "de",
    ["de_AT"] = "de",
    ["de_CH"] = "de",
    ["fr_FR"] = "fr",
    ["fr_CA"] = "fr",
    ["fr_CH"] = "fr",
    ["fr_BE"] = "fr",
    ["it_IT"] = "it",
    ["it_CH"] = "it",
    ["es_ES"] = "es",
    ["es_MX"] = "es",
    ["es_AR"] = "es",
    ["es_CO"] = "es",
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
    ["sr_RS"] = "sr",
    ["sl_SI"] = "sl",
    ["et_EE"] = "et",
    ["lv_LV"] = "lv",
    ["lt_LT"] = "lt",
    ["el_GR"] = "el",
    ["fi_FI"] = "fi",
    ["sv_SE"] = "sv",
    ["da_DK"] = "da",
    ["no_NO"] = "no",
    ["nb_NO"] = "no",
    ["nn_NO"] = "no",
    ["tr_TR"] = "tr",
    ["sq_AL"] = "sq",
    ["bs_BA"] = "bs",
    ["ca_ES"] = "ca",
    ["eu_ES"] = "eu",
    ["gl_ES"] = "gl",
    ["cy_GB"] = "cy",
    ["mt_MT"] = "mt",

    -- Asian and Middle Eastern languages
    ["ja_JP"] = "ja",
    ["ko_KR"] = "ko",
    ["zh_HK"] = "zh_HANT", -- Hong Kong usually uses Traditional
    ["zh_SG"] = "zh",      -- Singapore uses Simplified
    ["ar_SA"] = "ar",
    ["ar_EG"] = "ar",
    ["ar_AE"] = "ar",
    ["he_IL"] = "he",
    ["hi_IN"] = "hi",
    ["bn_BD"] = "bn",
    ["bn_IN"] = "bn",
    ["fa_IR"] = "fa",
    ["ur_PK"] = "ur",
    ["ta_IN"] = "ta",
    ["te_IN"] = "te",
    ["mr_IN"] = "mr",
    ["gu_IN"] = "gu",
    ["kn_IN"] = "kn",
    ["ml_IN"] = "ml",
    ["pa_IN"] = "pa",
    ["pa_PK"] = "pa",
    ["th_TH"] = "th",
    ["vi_VN"] = "vi",
    ["id_ID"] = "id",
    ["ms_MY"] = "ms",
    ["tl_PH"] = "tl",
    ["sw_KE"] = "sw",
    ["ne_NP"] = "ne",
    ["si_LK"] = "si",
    ["my_MM"] = "my", -- Burmese
    ["km_KH"] = "km", -- Khmer
    ["lo_LA"] = "lo", -- Lao

    -- Other supported languages
    ["af_ZA"] = "af",
    ["am_ET"] = "am",
    ["hy_AM"] = "hy",
    ["az_AZ"] = "az",
    ["be_BY"] = "be",
    ["ka_GE"] = "ka",
    ["kk_KZ"] = "kk",
    ["ky_KG"] = "ky",
    ["tg_TJ"] = "tg",
    ["tk_TM"] = "tk",
    ["uz_UZ"] = "uz",
    ["mn_MN"] = "mn",
}

babelfish.register_engine({
    translate = translate,
    language_codes = language_codes,
    language_alias = language_alias,
    mt_language_map = mt_language_map,

    compliance = nil, -- S("Translations are powered by Lingva"),
    engine_label = "Lingva Translate",
})
