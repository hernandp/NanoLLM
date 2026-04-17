// cli_shared.h : Shared command-line helper functions used by the app and self-tests.

#pragma once

#include "bpe.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using NamedArgumentMap = std::unordered_map<std::string, std::string>;

bool TryParseTargetSize(std::string_view text, std::size_t& value);
bool TryParseNamedArguments(int argc, char* argv[], int firstArgumentIndex, NamedArgumentMap& arguments);
bool ReadWholeFile(const std::string& path, std::string& contents);
bool ReadLines(const std::string& path, std::vector<std::string>& lines);
bool WriteVocabularyFile(const std::string& path, const BPEResult& result);
bool WriteMergeFile(const std::string& path, const BPEResult& result);
bool WriteTokenFile(const std::string& path, const std::vector<std::string>& tokens);
void PrintTokens(const std::vector<std::string>& tokens);
