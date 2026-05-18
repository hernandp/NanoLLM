// train.h : Minimal token-embedding trainer for the NanoLLM project.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Hyperparameters for the tiny embedding trainer.
// These defaults are intentionally small and readable rather than tuned.
struct TrainOptions
{
    std::size_t dimension = 64;
    std::size_t window = 4;
    std::size_t epochs = 3;
    std::size_t negativeSamples = 5;
    float learningRate = 0.025f;
    std::uint32_t seed = 1;
    bool showProgress = true;
};

// The in-memory result of training. The embedding matrix is stored row-major:
// embeddings[tokenId][component].
struct EmbeddingTrainingResult
{
    std::vector<std::string> vocabulary;
    std::vector<std::vector<float>> embeddings;
    std::size_t tokenCount = 0;
    std::size_t vocabularySizeUsed = 0;
};

bool LoadVocabularyFile(const std::string& path, std::vector<std::string>& vocabulary);

EmbeddingTrainingResult TrainEmbeddings(
    std::string_view corpus,
    const std::vector<std::string>& vocabulary,
    const std::vector<std::string>& mergeRules,
    const TrainOptions& options);
EmbeddingTrainingResult TrainEmbeddingsFromTokenIds(
    const std::vector<std::size_t>& tokenIds,
    const std::vector<std::string>& vocabulary,
    const TrainOptions& options);

bool WriteEmbeddingFile(const std::string& path, const EmbeddingTrainingResult& result);
bool WriteEmbeddingMetadataFile(
    const std::string& path,
    const EmbeddingTrainingResult& result,
    const TrainOptions& options,
    std::string_view tokenizerBase,
    std::string_view corpusPath,
    std::int64_t elapsedMilliseconds);
