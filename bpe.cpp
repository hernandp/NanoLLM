// bpe.cpp : A tiny Byte Pair Encoding (BPE) trainer for a toy LLM project.

#include "bpe.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Tokens to mark END of generation and padding.
constexpr const char* TOKEN_END = "<END>";
constexpr const char* TOKEN_PAD = "<PAD>";
constexpr const char* TOKEN_WORD_END = "</w>";

struct TokenPair
{
    std::string first;
    std::string second;

    bool operator==(const TokenPair& other) const = default;
};

struct TokenPairHash
{
    std::size_t operator()(const TokenPair& pair) const noexcept
    {
        const std::size_t h1 = std::hash<std::string>{}(pair.first);
        const std::size_t h2 = std::hash<std::string>{}(pair.second);
        return h1 ^ (h2 << 1);
    }
};

using TokenList = std::vector<std::string>;
using WordFrequencyMap = std::unordered_map<std::string, int>;

static std::vector<std::string> ExtractWords(std::string_view text)
{
    std::vector<std::string> words;
    std::string currentWord;

    for (const unsigned char ch : text)
    {
        if (std::isalpha(ch))
        {
            currentWord.push_back(static_cast<char>(std::tolower(ch)));
        }
        else if (!currentWord.empty())
        {
            words.push_back(currentWord);
            currentWord.clear();
        }
    }

    if (!currentWord.empty())
    {
        words.push_back(currentWord);
    }

    return words;
}

// Convert the corpus to lowercase words made of letters only.
// That keeps the toy example easy to follow.
static WordFrequencyMap BuildWordFrequencyMap(std::string_view corpus)
{
    WordFrequencyMap frequencies;
    for (const std::string& word : ExtractWords(corpus))
    {
        ++frequencies[word];
    }

    return frequencies;
}

// BPE starts from characters, so each word is split into 1-char tokens.
// We also append an end-of-word token to stop merges from crossing words.
static std::vector<TokenList> BuildInitialTokenizedWords(const WordFrequencyMap& wordFrequencies)
{
    std::vector<TokenList> tokenizedWords;
    tokenizedWords.reserve(wordFrequencies.size());

    for (const auto& [word, frequency] : wordFrequencies)
    {
        (void)frequency;

        TokenList tokens;
        tokens.reserve(word.size() + 1);

        for (const char ch : word)
        {
            tokens.emplace_back(1, ch);
        }

        tokens.emplace_back(TOKEN_WORD_END);
        tokenizedWords.push_back(std::move(tokens));
    }

    return tokenizedWords;
}

// Collect all current tokens into a de-duplicated vocabulary.
static std::vector<std::string> CollectVocabulary(const std::vector<TokenList>& tokenizedWords)
{
    std::vector<std::string> vocabulary;
    std::unordered_set<std::string> seen;

    vocabulary.emplace_back(TOKEN_PAD);
    vocabulary.emplace_back(TOKEN_END);
    seen.insert(vocabulary.back());
    seen.insert(vocabulary.front());

    for (const TokenList& word : tokenizedWords)
    {
        for (const std::string& token : word)
        {
            if (seen.insert(token).second)
            {
                vocabulary.push_back(token);
            }
        }
    }

    return vocabulary;
}

// Count how often every adjacent token pair appears.
// Word frequencies matter: pairs inside common words should be preferred.
static std::unordered_map<TokenPair, int, TokenPairHash> CountPairFrequencies(
    const std::vector<TokenList>& tokenizedWords,
    const WordFrequencyMap& wordFrequencies)
{
    std::unordered_map<TokenPair, int, TokenPairHash> pairFrequencies;

    std::size_t wordIndex = 0;
    for (const auto& [word, frequency] : wordFrequencies)
    {
        (void)word;
        const TokenList& tokens = tokenizedWords[wordIndex++];

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
        {
            const TokenPair pair{ tokens[i], tokens[i + 1] };
            pairFrequencies[pair] += frequency;
        }
    }

    return pairFrequencies;
}

// Replace every occurrence of the chosen pair in every tokenized word.
static void MergeTokenPair(std::vector<TokenList>& tokenizedWords, const TokenPair& pairToMerge)
{
    const std::string mergedToken = pairToMerge.first + pairToMerge.second;

    for (TokenList& tokens : tokenizedWords)
    {
        TokenList merged;
        merged.reserve(tokens.size());

        for (std::size_t i = 0; i < tokens.size();)
        {
            if (i + 1 < tokens.size() &&
                tokens[i] == pairToMerge.first &&
                tokens[i + 1] == pairToMerge.second)
            {
                merged.push_back(mergedToken);
                i += 2;
            }
            else
            {
                merged.push_back(tokens[i]);
                ++i;
            }
        }

        tokens = std::move(merged);
    }
}

static std::optional<TokenPair> ParseMergeRule(std::string_view mergeRule)
{
    const std::size_t plusPos = mergeRule.find(" + ");
    const std::size_t arrowPos = mergeRule.find(" -> ");

    if (plusPos == std::string_view::npos || arrowPos == std::string_view::npos || plusPos >= arrowPos)
    {
        return std::nullopt;
    }

    const std::string first(mergeRule.data(), plusPos);
    const std::string second(
        mergeRule.data() + plusPos + 3,
        arrowPos - (plusPos + 3));

    if (first.empty() || second.empty())
    {
        return std::nullopt;
    }

    return TokenPair{ first, second };
}

static TokenList TokenizeWord(std::string_view word)
{
    TokenList tokens;
    tokens.reserve(word.size() + 1);

    for (const char ch : word)
    {
        tokens.emplace_back(1, ch);
    }

    tokens.emplace_back(TOKEN_WORD_END);
    return tokens;
}

// Train a BPE vocabulary from an English text corpus.
// The result contains both the final vocabulary and the merge history.
BPEResult TrainBPE(std::string_view corpus, std::size_t targetVocabularySize)
{
    const WordFrequencyMap wordFrequencies = BuildWordFrequencyMap(corpus);
    std::vector<TokenList> tokenizedWords = BuildInitialTokenizedWords(wordFrequencies);
    std::vector<std::string> vocabulary = CollectVocabulary(tokenizedWords);
    std::vector<std::string> merges;

    while (vocabulary.size() < targetVocabularySize)
    {
        const auto pairFrequencies = CountPairFrequencies(tokenizedWords, wordFrequencies);
        if (pairFrequencies.empty())
        {
            break;
        }

        const auto bestPairIt = std::max_element(
            pairFrequencies.begin(),
            pairFrequencies.end(),
            [](const auto& left, const auto& right)
            {
                return left.second < right.second;
            });

        if (bestPairIt == pairFrequencies.end() || bestPairIt->second < 2)
        {
            break;
        }

        const TokenPair& bestPair = bestPairIt->first;
        const std::string mergedToken = bestPair.first + bestPair.second;

        MergeTokenPair(tokenizedWords, bestPair);
        vocabulary = CollectVocabulary(tokenizedWords);
        merges.push_back(bestPair.first + " + " + bestPair.second + " -> " + mergedToken);
    }

    return { vocabulary, merges };
}

std::vector<std::string> EncodeWithMerges(
    std::string_view text,
    const std::vector<std::string>& mergeRules)
{
    std::vector<std::string> encodedTokens;
    const std::vector<std::string> words = ExtractWords(text);

    for (const std::string& word : words)
    {
        TokenList tokens = TokenizeWord(word);

        // Apply merges in the same order they were learned during training.
        for (const std::string& mergeRule : mergeRules)
        {
            const std::optional<TokenPair> pair = ParseMergeRule(mergeRule);
            if (!pair.has_value())
            {
                continue;
            }

            std::vector<TokenList> wordAsSingleItem{ std::move(tokens) };
            MergeTokenPair(wordAsSingleItem, *pair);
            tokens = std::move(wordAsSingleItem.front());
        }

        for (std::string& token : tokens)
        {
            encodedTokens.push_back(std::move(token));
        }
    }

    return encodedTokens;
}
