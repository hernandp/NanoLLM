#include "linalg.h"
#include "bpe.h"
#include "cli_shared.h"
#include "selftest.h"
#include "train.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kLicenseText =
    "This is free and unencumbered software released into the public domain.\n"
    "\n"
    "Anyone is free to copy, modify, publish, use, compile, sell, or\n"
    "distribute this software, either in source code form or as a compiled\n"
    "binary, for any purpose, commercial or non-commercial, and by any\n"
    "means.\n"
    "\n"
    "In jurisdictions that recognize copyright laws, the author or authors\n"
    "of this software dedicate any and all copyright interest in the\n"
    "software to the public domain. We make this dedication for the benefit\n"
    "of the public at large and to the detriment of our heirs and\n"
    "successors. We intend this dedication to be an overt act of\n"
    "relinquishment in perpetuity of all present and future rights to this\n"
    "software under copyright law.\n"
    "\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND,\n"
    "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\n"
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n"
    "IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR\n"
    "OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,\n"
    "ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR\n"
    "OTHER DEALINGS IN THE SOFTWARE.\n"
    "\n"
    "For more information, please refer to <https://unlicense.org/>\n";

void PrintHelp(std::string_view programName)
{
    std::cout
        << "The Nano-LLM Experiment Project *** (c) 2026 Hernan Di Pietro\n\n"
        << "Available command line options:\n\n"
        << "/?                      Display this help\n"
        << "/lic                    Display license\n"
        << "/bpe src=s out=o m=n    Train vocabulary from a source corpus text file s,\n"
        << "                        using BPE algorithm with target size m, writes output to\n"
        << "                        file o.voc and o.merges. Example:\n"
        << "                       " << programName << " /bpe src=input.txt out=vocab m=1000\n"
        << "/encode mergefile=b     Encode text using merge file b.merges.\n"
        << "                        Optional input=i reads from file i, otherwise stdin is used\n"
        << "                        until EOF. Optional output=o writes to o.tok, otherwise the\n"
        << "                        encoded tokens are printed. Optional ids=t writes t.ids\n"
        << "                        with numeric token ids for faster training. Example:\n"
        << "                       " << programName
        << " /encode mergefile=vocab input=text.txt output=encoded ids=encoded\n"
        << "/train mergefile=b      Train token embeddings using tokenizer b.voc/b.merges.\n"
        << "       input=i          Source corpus text file, tokenized during training\n"
        << "       tokens=t         Existing t.ids token-id file, skips BPE tokenization\n"
        << "       output=o         Writes o.emb and o.meta\n"
        << "       [dim=64] [window=4] [epochs=3] [neg=5] [lr=0.025] [seed=1]\n"
        << "                        Example:\n"
        << "                       " << programName
        << " /train mergefile=vocab tokens=encoded output=model dim=64 epochs=3\n"
        << "/selftest               Run built-in tests for critical functionality\n";
}

bool TryParseTargetSize(std::string_view text, std::size_t& value)
{
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end && value > 0;
}

bool TryParseFloat(std::string_view text, float& value)
{
    try
    {
        std::size_t consumed = 0;
        value = std::stof(std::string(text), &consumed);
        return consumed == text.size();
    }
    catch (...)
    {
        return false;
    }
}

bool TryParseNamedArguments(int argc, char* argv[], int firstArgumentIndex, NamedArgumentMap& arguments)
{
    arguments.clear();

    for (int i = firstArgumentIndex; i < argc; ++i)
    {
        const std::string_view argument = argv[i];
        const std::size_t equalsPos = argument.find('=');

        if (equalsPos == std::string_view::npos || equalsPos == 0 || equalsPos + 1 >= argument.size())
        {
            return false;
        }

        const std::string key(argument.data(), equalsPos);
        const std::string value(argument.data() + equalsPos + 1, argument.size() - equalsPos - 1);

        if (!arguments.emplace(key, value).second)
        {
            return false;
        }
    }

    return true;
}

bool ReadWholeFile(const std::string& path, std::string& contents)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        return false;
    }

    contents.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

bool ReadLines(const std::string& path, std::vector<std::string>& lines)
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

bool ReadTokenIdFile(const std::string& path, std::vector<std::size_t>& tokenIds)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        return false;
    }

    tokenIds.clear();
    std::size_t tokenId = 0;
    while (input >> tokenId)
    {
        tokenIds.push_back(tokenId);
    }

    // If extraction stopped for something other than EOF, the file contains a
    // malformed token id. Treat that as a read failure instead of silently
    // training from a truncated stream.
    return input.eof();
}

bool WriteVocabularyFile(const std::string& path, const BPEResult& result)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    // Write one token per line so the output is easy to inspect by hand.
    for (const std::string& token : result.vocabulary)
    {
        output << token << '\n';
    }

    return static_cast<bool>(output);
}

bool WriteMergeFile(const std::string& path, const BPEResult& result)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    // Write one merge rule per line so the merge history is easy to inspect.
    for (const std::string& token : result.merges)
    {
        output << token << '\n';
    }

    return static_cast<bool>(output);
}

bool WriteTokenFile(const std::string& path, const std::vector<std::string>& tokens)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        if (i != 0)
        {
            output << ' ';
        }

        output << tokens[i];
    }

    output << '\n';
    return static_cast<bool>(output);
}

bool WriteTokenIdFile(const std::string& path, const std::vector<std::size_t>& tokenIds)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }

    // The .ids format is intentionally simple: whitespace-separated numeric
    // vocabulary row indexes. It is compact enough for toy experiments and easy
    // to inspect or generate by hand.
    for (std::size_t i = 0; i < tokenIds.size(); ++i)
    {
        if (i != 0)
        {
            output << ' ';
        }

        output << tokenIds[i];
    }

    output << '\n';
    return static_cast<bool>(output);
}

void PrintTokens(const std::vector<std::string>& tokens)
{
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ' ';
        }

        std::cout << tokens[i];
    }

    std::cout << '\n';
}

std::string WithDefaultExtension(const std::string& baseOrPath, std::string_view extension)
{
    if (baseOrPath.ends_with(extension))
    {
        return baseOrPath;
    }

    return baseOrPath + std::string(extension);
}

int main(int argc, char* argv[])
{
    const std::string_view programName = argc > 0 ? argv[0] : "nanollm";

    if (argc <= 1)
    {
        PrintHelp(programName);
        return 0;
    }

    const std::string_view command = argv[1];

    if (command == "/?" || command == "-?")
    {
        PrintHelp(programName);
        return 0;
    }

    if (command == "/lic")
    {
        std::cout << kLicenseText;
        return 0;
    }

    if (command == "/selftest")
    {
        return RunSelfTests();
    }

    if (command == "/bpe")
    {
        NamedArgumentMap arguments;
        const std::string bpeUsage =
            "Usage: " + std::string(programName) + " /bpe src=<file> out=<base-name> m=<target-size>\n";

        if (argc == 2 || !TryParseNamedArguments(argc, argv, 2, arguments))
        {
            std::cerr << bpeUsage;
            return 1;
        }

        const auto sourceIt = arguments.find("src");
        const auto outputIt = arguments.find("out");
        const auto targetSizeIt = arguments.find("m");

        if (sourceIt == arguments.end() || outputIt == arguments.end() || targetSizeIt == arguments.end())
        {
            std::cerr << "Missing required /bpe arguments. Expected src=..., out=..., and m=...\n";
            return 1;
        }

        if (arguments.size() != 3)
        {
            std::cerr << "Unknown /bpe argument detected. Expected only src=..., out=..., and m=...\n";
            return 1;
        }

        const std::string& sourcePath = sourceIt->second;
        const std::string& outputBasePath = outputIt->second;
        const std::string vocabularyPath = outputBasePath + ".voc";
        const std::string mergesPath = outputBasePath + ".merges";

        std::size_t targetVocabularySize = 0;
        if (!TryParseTargetSize(targetSizeIt->second, targetVocabularySize))
        {
            std::cerr << "Invalid target vocabulary size: " << targetSizeIt->second << '\n';
            return 1;
        }

        std::string corpus;
        if (!ReadWholeFile(sourcePath, corpus))
        {
            std::cerr << "Could not open source corpus file: " << sourcePath << '\n';
            return 1;
        }

        const auto startTime = std::chrono::steady_clock::now();
        const BPEResult result = TrainBPE(corpus, targetVocabularySize);
        const auto endTime = std::chrono::steady_clock::now();
        if (!WriteVocabularyFile(vocabularyPath, result))
        {
            std::cerr << "Could not write vocabulary file: " << vocabularyPath << '\n';
            return 1;
        }

        if (!WriteMergeFile(mergesPath, result))
        {
            std::cerr << "Could not write merge file: " << mergesPath << '\n';
            return 1;
        }

        const auto elapsedMilliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        std::cout << "Wrote " << result.vocabulary.size()
                  << " tokens to " << vocabularyPath
                  << " and " << result.merges.size()
                  << " merge rules to " << mergesPath
                  << " in " << elapsedMilliseconds.count() << " ms\n";
        return 0;
    }

    if (command == "/encode")
    {
        NamedArgumentMap arguments;
        const std::string encodeUsage =
            "Usage: " + std::string(programName) +
            " /encode mergefile=<base-name> [input=<file>] [output=<base-name>] [ids=<base-name>]\n";

        if (argc == 2 || !TryParseNamedArguments(argc, argv, 2, arguments))
        {
            std::cerr << encodeUsage;
            return 1;
        }

        const auto mergeFileIt = arguments.find("mergefile");
        if (mergeFileIt == arguments.end())
        {
            std::cerr << "Missing required /encode argument: mergefile=...\n";
            return 1;
        }

        if (arguments.size() < 1 || arguments.size() > 4)
        {
            std::cerr << "Unknown /encode argument detected. Expected mergefile=..., optional input=..., optional output=..., optional ids=...\n";
            return 1;
        }

        for (const auto& [key, value] : arguments)
        {
            (void)value;
            if (key != "mergefile" && key != "input" && key != "output" && key != "ids")
            {
                std::cerr << "Unknown /encode argument: " << key << '\n';
                return 1;
            }
        }

        const std::string mergePath = mergeFileIt->second + ".merges";
        std::vector<std::string> mergeRules;
        if (!ReadLines(mergePath, mergeRules))
        {
            std::cerr << "Could not open merge file: " << mergePath << '\n';
            return 1;
        }

        std::string inputText;
        const auto inputIt = arguments.find("input");
        if (inputIt != arguments.end())
        {
            if (!ReadWholeFile(inputIt->second, inputText))
            {
                std::cerr << "Could not open input file: " << inputIt->second << '\n';
                return 1;
            }
        }
        else
        {
            inputText.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
        }

        const auto startTime = std::chrono::steady_clock::now();
        const auto encodedTokens = EncodeWithMerges(inputText, mergeRules);
        std::vector<std::size_t> encodedTokenIds;
        const auto idsIt = arguments.find("ids");
        if (idsIt != arguments.end())
        {
            std::vector<std::string> vocabulary;
            const std::string vocabularyPath = mergeFileIt->second + ".voc";
            if (!LoadVocabularyFile(vocabularyPath, vocabulary))
            {
                std::cerr << "Could not open vocabulary file: " << vocabularyPath << '\n';
                return 1;
            }

            encodedTokenIds = TokenStringsToIds(encodedTokens, vocabulary);
        }

        const auto endTime = std::chrono::steady_clock::now();
        const auto elapsedMilliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        const auto outputIt = arguments.find("output");
        if (outputIt != arguments.end())
        {
            const std::string outputPath = outputIt->second + ".tok";
            if (!WriteTokenFile(outputPath, encodedTokens))
            {
                std::cerr << "Could not write token file: " << outputPath << '\n';
                return 1;
            }

            std::cout << "Wrote " << encodedTokens.size()
                      << " encoded tokens to " << outputPath
                      << " in " << elapsedMilliseconds.count() << " ms\n";
        }
        else
        {
            PrintTokens(encodedTokens);
            std::cout << "Generated " << encodedTokens.size()
                      << " encoded tokens in " << elapsedMilliseconds.count() << " ms\n";
        }

        if (idsIt != arguments.end())
        {
            const std::string idsPath = WithDefaultExtension(idsIt->second, ".ids");
            if (!WriteTokenIdFile(idsPath, encodedTokenIds))
            {
                std::cerr << "Could not write token id file: " << idsPath << '\n';
                return 1;
            }

            std::cout << "Wrote " << encodedTokenIds.size()
                      << " token ids to " << idsPath << '\n';
        }

        return 0;
    }

    if (command == "/train")
    {
        NamedArgumentMap arguments;
        const std::string trainUsage =
            "Usage: " + std::string(programName) +
            " /train mergefile=<base-name> output=<base-name> (input=<file> | tokens=<base-name-or-file>)"
            " [dim=<n>] [window=<n>] [epochs=<n>] [neg=<n>] [lr=<f>] [seed=<n>]\n";

        if (argc == 2 || !TryParseNamedArguments(argc, argv, 2, arguments))
        {
            std::cerr << trainUsage;
            return 1;
        }

        const auto mergeFileIt = arguments.find("mergefile");
        const auto inputIt = arguments.find("input");
        const auto tokensIt = arguments.find("tokens");
        const auto outputIt = arguments.find("output");

        if (mergeFileIt == arguments.end() || outputIt == arguments.end() ||
            (inputIt == arguments.end() && tokensIt == arguments.end()))
        {
            std::cerr << trainUsage;
            return 1;
        }

        if (inputIt != arguments.end() && tokensIt != arguments.end())
        {
            std::cerr << "Use either input=... or tokens=... for /train, not both.\n";
            return 1;
        }

        for (const auto& [key, value] : arguments)
        {
            (void)value;
            if (key != "mergefile" && key != "input" && key != "tokens" && key != "output" &&
                key != "dim" && key != "window" && key != "epochs" &&
                key != "neg" && key != "lr" && key != "seed")
            {
                std::cerr << "Unknown /train argument: " << key << '\n';
                return 1;
            }
        }

        TrainOptions options;

        if (const auto it = arguments.find("dim"); it != arguments.end() &&
            !TryParseTargetSize(it->second, options.dimension))
        {
            std::cerr << "Invalid embedding dimension: " << it->second << '\n';
            return 1;
        }

        if (const auto it = arguments.find("window"); it != arguments.end() &&
            !TryParseTargetSize(it->second, options.window))
        {
            std::cerr << "Invalid context window: " << it->second << '\n';
            return 1;
        }

        if (const auto it = arguments.find("epochs"); it != arguments.end() &&
            !TryParseTargetSize(it->second, options.epochs))
        {
            std::cerr << "Invalid epoch count: " << it->second << '\n';
            return 1;
        }

        if (const auto it = arguments.find("neg"); it != arguments.end() &&
            !TryParseTargetSize(it->second, options.negativeSamples))
        {
            std::cerr << "Invalid negative sample count: " << it->second << '\n';
            return 1;
        }

        if (const auto it = arguments.find("lr"); it != arguments.end() &&
            !TryParseFloat(it->second, options.learningRate))
        {
            std::cerr << "Invalid learning rate: " << it->second << '\n';
            return 1;
        }

        if (options.learningRate <= 0.0f)
        {
            std::cerr << "Learning rate must be positive\n";
            return 1;
        }

        if (const auto it = arguments.find("seed"); it != arguments.end())
        {
            std::size_t parsedSeed = 0;
            if (!TryParseTargetSize(it->second, parsedSeed))
            {
                std::cerr << "Invalid seed: " << it->second << '\n';
                return 1;
            }

            options.seed = static_cast<std::uint32_t>(parsedSeed);
        }

        const std::string tokenizerBase = mergeFileIt->second;
        const std::string vocabularyPath = tokenizerBase + ".voc";
        const std::string mergePath = tokenizerBase + ".merges";
        const std::string embeddingPath = outputIt->second + ".emb";
        const std::string metadataPath = outputIt->second + ".meta";

        std::vector<std::string> vocabulary;
        if (!LoadVocabularyFile(vocabularyPath, vocabulary))
        {
            std::cerr << "Could not open vocabulary file: " << vocabularyPath << '\n';
            return 1;
        }

        const auto startTime = std::chrono::steady_clock::now();
        EmbeddingTrainingResult result;
        std::string trainingSourcePath;

        if (tokensIt != arguments.end())
        {
            std::vector<std::size_t> tokenIds;
            trainingSourcePath = WithDefaultExtension(tokensIt->second, ".ids");
            if (!ReadTokenIdFile(trainingSourcePath, tokenIds))
            {
                std::cerr << "Could not read token id file: " << trainingSourcePath << '\n';
                return 1;
            }

            result = TrainEmbeddingsFromTokenIds(tokenIds, vocabulary, options);
        }
        else
        {
            std::vector<std::string> mergeRules;
            if (!ReadLines(mergePath, mergeRules))
            {
                std::cerr << "Could not open merge file: " << mergePath << '\n';
                return 1;
            }

            std::string corpus;
            trainingSourcePath = inputIt->second;
            if (!ReadWholeFile(trainingSourcePath, corpus))
            {
                std::cerr << "Could not open source corpus file: " << trainingSourcePath << '\n';
                return 1;
            }

            result = TrainEmbeddings(corpus, vocabulary, mergeRules, options);
        }

        const auto endTime = std::chrono::steady_clock::now();
        const auto elapsedMilliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        if (!WriteEmbeddingFile(embeddingPath, result))
        {
            std::cerr << "Could not write embedding file: " << embeddingPath << '\n';
            return 1;
        }

        if (!WriteEmbeddingMetadataFile(
                metadataPath,
                result,
                options,
                tokenizerBase,
                trainingSourcePath,
                elapsedMilliseconds))
        {
            std::cerr << "Could not write embedding metadata file: " << metadataPath << '\n';
            return 1;
        }

        std::cout << "Trained " << result.vocabularySizeUsed
                  << " token embeddings from " << result.tokenCount
                  << " encoded training tokens across " << options.epochs
                  << " epochs, wrote " << embeddingPath
                  << " and " << metadataPath
                  << " in " << elapsedMilliseconds << " ms\n";
        return 0;
    }

    std::cerr << "Unknown command: " << command << "\n\n";
    PrintHelp(programName);
    return 1;
}
