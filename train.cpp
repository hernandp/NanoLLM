// train.cpp : Minimal embedding trainer for a nano-sized educational LLM.
//
// This file intentionally favors clarity over performance. The goal is to show
// the moving parts of a tiny word2vec-style trainer in code that is easy to read.

#include "train.h"

#include "bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
using TokenId = std::size_t;

// Numeric constants that make the optimizer stable in this tiny implementation.
constexpr float kSigmoidClamp = 8.0f;
constexpr float kNegativeSamplePower = 0.75f;

// Parse one-token-per-line vocabulary files written by /bpe.
bool ReadNonEmptyLines(const std::string& path, std::vector<std::string>& lines)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        return false;
    }

    lines.clear();
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (!line.empty())
        {
            lines.push_back(line);
        }
    }

    return true;
}

// The exported embedding file contains a dense vector per token. We store the
// training tables in a flat array because it keeps the math straightforward and
// avoids nested allocations during SGD updates.
using FlatMatrix = std::vector<float>;

std::size_t RowOffset(std::size_t row, std::size_t width)
{
    return row * width;
}

// This is a numerically safe sigmoid for the tiny trainer. Extreme scores are
// clamped because exp(large_number) is not useful for educational SGD and only
// risks overflow.
float Sigmoid(float x)
{
    x = std::clamp(x, -kSigmoidClamp, kSigmoidClamp);
    return 1.0f / (1.0f + std::exp(-x));
}

// Compute the dot product between one row from the input embedding matrix and
// one row from the output/context matrix.
float DotRow(
    const FlatMatrix& left,
    const FlatMatrix& right,
    std::size_t leftRow,
    std::size_t rightRow,
    std::size_t width)
{
    const std::size_t leftOffset = RowOffset(leftRow, width);
    const std::size_t rightOffset = RowOffset(rightRow, width);

    float sum = 0.0f;
    for (std::size_t i = 0; i < width; ++i)
    {
        sum += left[leftOffset + i] * right[rightOffset + i];
    }

    return sum;
}

// Apply one skip-gram update. In skip-gram we try to make the center token
// predict nearby context tokens. Negative sampling turns that into a sequence of
// binary classification updates:
//   - label 1 for the true context token
//   - label 0 for sampled "noise" tokens
//
// We keep two tables during training:
//   1. inputEmbeddings  : the vector for the center token
//   2. outputEmbeddings : the vector for the predicted context token
//
// This mirrors classic word2vec training. At the end we export only the input
// table because it is the one usually used as the learned embedding space.
void ApplyBinaryUpdate(
    FlatMatrix& inputEmbeddings,
    FlatMatrix& outputEmbeddings,
    std::size_t centerId,
    std::size_t targetId,
    float label,
    std::size_t width,
    float learningRate)
{
    const std::size_t inputOffset = RowOffset(centerId, width);
    const std::size_t outputOffset = RowOffset(targetId, width);

    const float score = DotRow(inputEmbeddings, outputEmbeddings, centerId, targetId, width);
    const float prediction = Sigmoid(score);
    const float error = (label - prediction) * learningRate;

    // The center row and context row are both updated. We cache the original
    // center vector because both gradients depend on the pre-update values.
    std::vector<float> inputSnapshot(width);
    for (std::size_t i = 0; i < width; ++i)
    {
        inputSnapshot[i] = inputEmbeddings[inputOffset + i];
    }

    for (std::size_t i = 0; i < width; ++i)
    {
        inputEmbeddings[inputOffset + i] += error * outputEmbeddings[outputOffset + i];
        outputEmbeddings[outputOffset + i] += error * inputSnapshot[i];
    }
}

// Count token ids so we can build the negative-sampling distribution. Reusing
// ids from a .ids file avoids running the BPE encoder again during /train.
std::vector<std::size_t> CountTokenIds(
    const std::vector<TokenId>& tokenIds,
    std::size_t vocabularySize)
{
    std::vector<std::size_t> tokenCounts(vocabularySize, 0);

    for (TokenId tokenId : tokenIds)
    {
        if (tokenId >= vocabularySize)
        {
            continue;
        }

        ++tokenCounts[tokenId];
    }

    return tokenCounts;
}

// word2vec-style negative sampling does not sample uniformly. Frequent tokens
// remain likely, but the exponent 0.75 reduces the dominance of the very most
// common items.
std::vector<double> BuildNegativeSamplingWeights(const std::vector<std::size_t>& tokenCounts)
{
    std::vector<double> weights(tokenCounts.size(), 0.0);

    for (std::size_t i = 0; i < tokenCounts.size(); ++i)
    {
        if (tokenCounts[i] == 0)
        {
            continue;
        }

        weights[i] = std::pow(static_cast<double>(tokenCounts[i]), kNegativeSamplePower);
    }

    return weights;
}

// Convert the flat training matrix into a friendlier row-per-token structure for
// writing and later inspection.
std::vector<std::vector<float>> UnflattenEmbeddings(
    const FlatMatrix& flat,
    std::size_t rows,
    std::size_t width)
{
    std::vector<std::vector<float>> result(rows, std::vector<float>(width));

    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t offset = RowOffset(row, width);
        for (std::size_t i = 0; i < width; ++i)
        {
            result[row][i] = flat[offset + i];
        }
    }

    return result;
}

// Print coarse-grained progress so long training runs do not look stalled.
// We update in chunks rather than every token because console IO is much slower
// than the math itself and would otherwise dominate the runtime in this tiny trainer.
void PrintTrainingProgress(
    std::size_t epochIndex,
    std::size_t epochCount,
    std::size_t completedCenters,
    std::size_t totalCenters,
    const std::chrono::steady_clock::time_point& startTime)
{
    if (totalCenters == 0)
    {
        return;
    }

    const std::size_t percent = (completedCenters * 100) / totalCenters;
    const auto elapsedMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();

    std::cout << '\r'
              << "Epoch " << (epochIndex + 1) << '/' << epochCount
              << " - " << percent << "% - "
              << completedCenters << '/' << totalCenters
              << " center tokens - " << elapsedMilliseconds << " ms elapsed"
              << std::flush;
}
}

bool LoadVocabularyFile(const std::string& path, std::vector<std::string>& vocabulary)
{
    return ReadNonEmptyLines(path, vocabulary);
}

EmbeddingTrainingResult TrainEmbeddings(
    std::string_view corpus,
    const std::vector<std::string>& vocabulary,
    const std::vector<std::string>& mergeRules,
    const TrainOptions& options)
{
    // Raw-text training is a convenience wrapper: tokenize once with the saved
    // BPE rules, then train from the resulting numeric ids. The /train tokens=
    // path below skips this encoding step by loading ids from disk directly.
    const std::vector<TokenId> tokenIds = EncodeWithMergesToIds(corpus, mergeRules, vocabulary);
    return TrainEmbeddingsFromTokenIds(tokenIds, vocabulary, options);
}

EmbeddingTrainingResult TrainEmbeddingsFromTokenIds(
    const std::vector<std::size_t>& tokenIds,
    const std::vector<std::string>& vocabulary,
    const TrainOptions& options)
{
    EmbeddingTrainingResult result;
    result.vocabulary = vocabulary;
    result.vocabularySizeUsed = vocabulary.size();

    if (vocabulary.empty() || options.dimension == 0)
    {
        return result;
    }

    // Step 1: accept the already-encoded token id stream. This is the compact
    // representation the trainer actually needs: each number is a row index in
    // the vocabulary/embedding table. Invalid ids are ignored so a hand-edited
    // .ids file cannot make the trainer index past the embedding matrix.
    std::vector<TokenId> trainingTokenIds;
    trainingTokenIds.reserve(tokenIds.size());
    for (TokenId tokenId : tokenIds)
    {
        if (tokenId < vocabulary.size())
        {
            trainingTokenIds.push_back(tokenId);
        }
    }

    result.tokenCount = trainingTokenIds.size();

    if (trainingTokenIds.empty())
    {
        result.embeddings.assign(vocabulary.size(), std::vector<float>(options.dimension, 0.0f));
        return result;
    }

    // Step 2: initialize the two training tables with tiny random values.
    // Small numbers keep the initial scores near zero, where the sigmoid has a
    // useful gradient.
    std::mt19937 rng(options.seed);
    std::uniform_real_distribution<float> initDist(
        -0.5f / static_cast<float>(options.dimension),
         0.5f / static_cast<float>(options.dimension));

    FlatMatrix inputEmbeddings(vocabulary.size() * options.dimension, 0.0f);
    FlatMatrix outputEmbeddings(vocabulary.size() * options.dimension, 0.0f);

    for (float& value : inputEmbeddings)
    {
        value = initDist(rng);
    }

    for (float& value : outputEmbeddings)
    {
        value = initDist(rng);
    }

    // Step 3: create the negative-sampling distribution from token counts.
    const std::vector<std::size_t> tokenCounts = CountTokenIds(trainingTokenIds, vocabulary.size());
    const std::vector<double> negativeWeights = BuildNegativeSamplingWeights(tokenCounts);
    std::discrete_distribution<std::size_t> negativeSampler(
        negativeWeights.begin(),
        negativeWeights.end());

    // Step 4: walk the token stream and generate skip-gram training pairs.
    // For each center token, nearby tokens inside the window are positive
    // examples. Each positive example is paired with several random negatives.
    const auto trainingStartTime = std::chrono::steady_clock::now();
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch)
    {
        if (options.showProgress)
        {
            std::cout << "Starting epoch " << (epoch + 1)
                      << '/' << options.epochs << '\n';
        }

        const std::size_t progressStep = std::max<std::size_t>(1, trainingTokenIds.size() / 100);
        for (std::size_t centerIndex = 0; centerIndex < trainingTokenIds.size(); ++centerIndex)
        {
            const TokenId centerId = trainingTokenIds[centerIndex];

            const std::size_t begin =
                (centerIndex > options.window) ? centerIndex - options.window : 0;
            const std::size_t end =
                std::min(trainingTokenIds.size(), centerIndex + options.window + 1);

            for (std::size_t contextIndex = begin; contextIndex < end; ++contextIndex)
            {
                if (contextIndex == centerIndex)
                {
                    continue;
                }

                const TokenId contextId = trainingTokenIds[contextIndex];

                // Positive pair: center token should predict the real context.
                ApplyBinaryUpdate(
                    inputEmbeddings,
                    outputEmbeddings,
                    centerId,
                    contextId,
                    1.0f,
                    options.dimension,
                    options.learningRate);

                // Negative pairs: center token should reject randomly sampled
                // noise tokens.
                for (std::size_t negativeIndex = 0; negativeIndex < options.negativeSamples; ++negativeIndex)
                {
                    std::size_t negativeId = negativeSampler(rng);

                    // Sampling the true context token as a "negative" weakens
                    // the learning signal, so skip it and draw again.
                    while (negativeId == contextId && vocabulary.size() > 1)
                    {
                        negativeId = negativeSampler(rng);
                    }

                    ApplyBinaryUpdate(
                        inputEmbeddings,
                        outputEmbeddings,
                        centerId,
                        negativeId,
                        0.0f,
                        options.dimension,
                        options.learningRate);
                }
            }

            if (options.showProgress)
            {
                const std::size_t completedCenters = centerIndex + 1;
                if (completedCenters == trainingTokenIds.size() ||
                    (completedCenters % progressStep) == 0)
                {
                    PrintTrainingProgress(
                        epoch,
                        options.epochs,
                        completedCenters,
                        trainingTokenIds.size(),
                        trainingStartTime);
                }
            }
        }

        if (options.showProgress)
        {
            std::cout << '\n';
        }
    }

    // Only the input embedding table is exported. That is the standard
    // simplified representation used as the learned token embedding space.
    result.embeddings = UnflattenEmbeddings(inputEmbeddings, vocabulary.size(), options.dimension);
    return result;
}

bool WriteEmbeddingFile(const std::string& path, const EmbeddingTrainingResult& result)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    if (result.embeddings.empty())
    {
        output << "tokens=" << result.vocabulary.size() << " dim=0\n";
        return static_cast<bool>(output);
    }

    output << "tokens=" << result.vocabulary.size()
           << " dim=" << result.embeddings.front().size() << '\n';

    for (std::size_t tokenId = 0; tokenId < result.vocabulary.size(); ++tokenId)
    {
        output << result.vocabulary[tokenId];

        for (float value : result.embeddings[tokenId])
        {
            output << '\t' << value;
        }

        output << '\n';
    }

    return static_cast<bool>(output);
}

bool WriteEmbeddingMetadataFile(
    const std::string& path,
    const EmbeddingTrainingResult& result,
    const TrainOptions& options,
    std::string_view tokenizerBase,
    std::string_view corpusPath,
    std::int64_t elapsedMilliseconds)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    output << "tokenizer=" << tokenizerBase << '\n';
    output << "corpus=" << corpusPath << '\n';
    output << "dim=" << options.dimension << '\n';
    output << "window=" << options.window << '\n';
    output << "epochs=" << options.epochs << '\n';
    output << "neg=" << options.negativeSamples << '\n';
    output << "lr=" << options.learningRate << '\n';
    output << "seed=" << options.seed << '\n';
    output << "token_count=" << result.tokenCount << '\n';
    output << "vocabulary_size=" << result.vocabularySizeUsed << '\n';
    output << "elapsed_ms=" << elapsedMilliseconds << '\n';

    return static_cast<bool>(output);
}
