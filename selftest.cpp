// selftest.cpp : In-process self-tests for critical NanoLLM functionality.

#include "selftest.h"

#include "bpe.h"
#include "cli_shared.h"
#include "train.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
struct SelfTestContext
{
    int passed = 0;
    int failed = 0;

    void Check(bool condition, std::string_view testName, std::string_view failureMessage)
    {
        if (condition)
        {
            ++passed;
            std::cout << "[PASS] " << testName << '\n';
        }
        else
        {
            ++failed;
            std::cerr << "[FAIL] " << testName << ": " << failureMessage << '\n';
        }
    }
};

void RemoveIfExists(const std::string& path)
{
    (void)std::remove(path.c_str());
}
}

int RunSelfTests()
{
    SelfTestContext tests;

    {
        char arg0[] = "nanollm";
        char arg1[] = "/bpe";
        char arg2[] = "src=corpus.txt";
        char arg3[] = "out=output";
        char arg4[] = "m=128";
        char* argv[] = { arg0, arg1, arg2, arg3, arg4 };

        NamedArgumentMap arguments;
        const bool parsed = TryParseNamedArguments(5, argv, 2, arguments);

        tests.Check(parsed, "ParseNamedArguments.AcceptsKeyValueForm", "Expected src=, out= and m= to parse");
        tests.Check(arguments["src"] == "corpus.txt", "ParseNamedArguments.ReadsSource", "src value mismatch");
        tests.Check(arguments["out"] == "output", "ParseNamedArguments.ReadsOutput", "out value mismatch");
        tests.Check(arguments["m"] == "128", "ParseNamedArguments.ReadsTargetSize", "m value mismatch");
    }

    {
        char arg0[] = "nanollm";
        char arg1[] = "/bpe";
        char arg2[] = "src=corpus.txt";
        char arg3[] = "broken";
        char* argv[] = { arg0, arg1, arg2, arg3 };

        NamedArgumentMap arguments;
        const bool parsed = TryParseNamedArguments(4, argv, 2, arguments);
        tests.Check(!parsed, "ParseNamedArguments.RejectsMalformedEntry", "Malformed key/value input should fail");
    }

    const std::string corpus = "low lower lowest low lower";
    const BPEResult trained = TrainBPE(corpus, 32);

    tests.Check(!trained.vocabulary.empty(), "TrainBPE.BuildsVocabulary", "Expected a non-empty vocabulary");
    tests.Check(!trained.merges.empty(), "TrainBPE.BuildsMergeRules", "Expected at least one merge rule");

    const auto containsToken = [&trained](std::string_view token)
    {
        for (const std::string& current : trained.vocabulary)
        {
            if (current == token)
            {
                return true;
            }
        }

        return false;
    };

    tests.Check(containsToken("<PAD>"), "TrainBPE.AddsPadToken", "Vocabulary is missing <PAD>");
    tests.Check(containsToken("<END>"), "TrainBPE.AddsEndToken", "Vocabulary is missing <END>");

    const std::vector<std::string> encoded = EncodeWithMerges("lowest lower", trained.merges);
    tests.Check(!encoded.empty(), "EncodeWithMerges.ProducesTokens", "Expected encoded output");

    const std::size_t naiveTokenCount =
        std::string_view("lowest").size() + 1 +
        std::string_view("lower").size() + 1;
    tests.Check(
        encoded.size() < naiveTokenCount,
        "EncodeWithMerges.AppliesMerges",
        "Encoding should reduce token count compared to plain character tokens");

    const std::string basePath = "selftest_tmp";
    const std::string vocabularyPath = basePath + ".voc";
    const std::string mergesPath = basePath + ".merges";
    const std::string tokensPath = basePath + ".tok";
    const std::string tokenIdsPath = basePath + ".ids";
    const std::string embeddingPath = basePath + ".emb";
    const std::string metadataPath = basePath + ".meta";

    RemoveIfExists(vocabularyPath);
    RemoveIfExists(mergesPath);
    RemoveIfExists(tokensPath);
    RemoveIfExists(tokenIdsPath);
    RemoveIfExists(embeddingPath);
    RemoveIfExists(metadataPath);

    const std::vector<std::size_t> encodedTokenIds =
        EncodeWithMergesToIds("lowest lower", trained.merges, trained.vocabulary);

    tests.Check(WriteVocabularyFile(vocabularyPath, trained), "WriteVocabularyFile.WritesFile", "Could not write vocabulary file");
    tests.Check(WriteMergeFile(mergesPath, trained), "WriteMergeFile.WritesFile", "Could not write merge file");
    tests.Check(WriteTokenFile(tokensPath, encoded), "WriteTokenFile.WritesFile", "Could not write token file");
    tests.Check(WriteTokenIdFile(tokenIdsPath, encodedTokenIds), "WriteTokenIdFile.WritesFile", "Could not write token id file");

    std::vector<std::string> mergeLines;
    std::vector<std::size_t> reloadedTokenIds;
    std::string tokenFileContents;
    tests.Check(ReadLines(mergesPath, mergeLines), "ReadLines.ReadsMergeFile", "Could not read merge file back");
    tests.Check(ReadWholeFile(tokensPath, tokenFileContents), "ReadWholeFile.ReadsTokenFile", "Could not read token file back");
    tests.Check(ReadTokenIdFile(tokenIdsPath, reloadedTokenIds), "ReadTokenIdFile.ReadsTokenIdFile", "Could not read token id file back");
    tests.Check(mergeLines.size() == trained.merges.size(), "ReadLines.PreservesMergeCount", "Merge count changed after file round-trip");
    tests.Check(tokenFileContents.find(encoded.front()) != std::string::npos, "WriteTokenFile.PreservesTokenText", "Token file did not contain encoded tokens");
    tests.Check(!encodedTokenIds.empty(), "EncodeWithMergesToIds.ProducesIds", "Expected token ids from encoded text");
    tests.Check(reloadedTokenIds == encodedTokenIds, "ReadTokenIdFile.PreservesIds", "Token id file changed after round-trip");

    std::vector<std::string> reloadedVocabulary;
    tests.Check(LoadVocabularyFile(vocabularyPath, reloadedVocabulary), "LoadVocabularyFile.ReadsVocabulary", "Could not read vocabulary file back");
    tests.Check(reloadedVocabulary == trained.vocabulary, "LoadVocabularyFile.PreservesVocabularyOrder", "Vocabulary file changed token order");

    TrainOptions options;
    options.dimension = 8;
    options.window = 2;
    options.epochs = 1;
    options.negativeSamples = 2;
    options.learningRate = 0.05f;
    options.seed = 123;
    options.showProgress = false;

    const EmbeddingTrainingResult trainingRun1 =
        TrainEmbeddings(corpus, trained.vocabulary, trained.merges, options);
    const EmbeddingTrainingResult trainingRun2 =
        TrainEmbeddings(corpus, trained.vocabulary, trained.merges, options);
    const std::vector<std::size_t> corpusTokenIds =
        EncodeWithMergesToIds(corpus, trained.merges, trained.vocabulary);
    const EmbeddingTrainingResult trainingFromIds =
        TrainEmbeddingsFromTokenIds(corpusTokenIds, trained.vocabulary, options);

    tests.Check(trainingRun1.tokenCount > 0, "TrainEmbeddings.CountsTokens", "Expected encoded training tokens");
    tests.Check(trainingRun1.vocabulary.size() == trained.vocabulary.size(), "TrainEmbeddings.KeepsVocabulary", "Embedding trainer changed vocabulary size");
    tests.Check(trainingRun1.embeddings.size() == trained.vocabulary.size(), "TrainEmbeddings.ProducesRowsPerToken", "Embedding row count mismatch");
    tests.Check(!trainingRun1.embeddings.empty() && trainingRun1.embeddings.front().size() == options.dimension, "TrainEmbeddings.UsesRequestedDimension", "Embedding dimension mismatch");
    tests.Check(trainingRun1.embeddings == trainingRun2.embeddings, "TrainEmbeddings.IsDeterministicWithFixedSeed", "Repeated runs with same seed should match");
    tests.Check(trainingRun1.embeddings == trainingFromIds.embeddings, "TrainEmbeddingsFromTokenIds.MatchesRawTextPath", "Training from pre-tokenized ids should match raw text path");

    tests.Check(WriteEmbeddingFile(embeddingPath, trainingRun1), "WriteEmbeddingFile.WritesFile", "Could not write embedding file");
    tests.Check(
        WriteEmbeddingMetadataFile(metadataPath, trainingRun1, options, "selftest_tmp", "corpus.txt", 42),
        "WriteEmbeddingMetadataFile.WritesFile",
        "Could not write embedding metadata file");

    std::string embeddingFileContents;
    std::string metadataFileContents;
    tests.Check(ReadWholeFile(embeddingPath, embeddingFileContents), "ReadWholeFile.ReadsEmbeddingFile", "Could not read embedding file back");
    tests.Check(ReadWholeFile(metadataPath, metadataFileContents), "ReadWholeFile.ReadsEmbeddingMetadataFile", "Could not read embedding metadata file back");
    tests.Check(
        embeddingFileContents.find("tokens=") == 0,
        "WriteEmbeddingFile.WritesHeader",
        "Embedding file header is missing");
    tests.Check(
        metadataFileContents.find("dim=8") != std::string::npos,
        "WriteEmbeddingMetadataFile.WritesHyperparameters",
        "Embedding metadata file is missing hyperparameters");

    RemoveIfExists(vocabularyPath);
    RemoveIfExists(mergesPath);
    RemoveIfExists(tokensPath);
    RemoveIfExists(tokenIdsPath);
    RemoveIfExists(embeddingPath);
    RemoveIfExists(metadataPath);

    std::cout << "\nSelf-test summary: " << tests.passed << " passed, " << tests.failed << " failed\n";
    return tests.failed == 0 ? 0 : 1;
}
