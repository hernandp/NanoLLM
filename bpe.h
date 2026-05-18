// bpe.h : Simple Byte Pair Encoding (BPE) trainer for a toy LLM project.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

struct BPEResult
{
    std::vector<std::string> vocabulary;
    std::vector<std::string> merges;
};

BPEResult TrainBPE(std::string_view corpus, std::size_t targetVocabularySize);
std::vector<std::string> EncodeWithMerges(
    std::string_view text,
    const std::vector<std::string>& mergeRules);
std::vector<std::size_t> TokenStringsToIds(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& vocabulary);
std::vector<std::size_t> EncodeWithMergesToIds(
    std::string_view text,
    const std::vector<std::string>& mergeRules,
    const std::vector<std::string>& vocabulary);
