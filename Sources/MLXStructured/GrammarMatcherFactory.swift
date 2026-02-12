//
//  GrammarMatcherFactory.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 20.09.2025.
//

import MLXLMCommon
import Hub

public extension GrammarMaskedLogitProcessor {
    static func from(
        hub: HubApi = .shared, // TODO: Request changes in swift-transformers to make the tokenizer vocab (and some other properties) public
        configuration: ModelConfiguration,
        grammar: Grammar
    ) async throws -> GrammarMaskedLogitProcessor {
        let configurations = switch configuration.id {
        case .id(let id, let revision):
            LanguageModelConfigurationFromHub(modelName: id, revision: revision, hubApi: hub)
        case .directory(let directory):
            LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
        }
        
        let (modelConfig, tokenizerConfig, tokenizerData) = try await (
            configurations.modelConfig,
            configurations.tokenizerConfig,
            configurations.tokenizerData
        )
        
        // Compute vocab size from both model config and actual tokenizer indices so we never
        // index out of range when the tokenizer has more entries (e.g. added tokens) than
        // modelConfig.vocabSize.
        var maxIndex = modelConfig?.vocabSize.integer().map { $0 - 1 } ?? -1
        for (_, value) in tokenizerData.model.vocab.dictionary(or: [:]) {
            if let index = value.integer(), index > maxIndex {
                maxIndex = index
            }
        }
        for value in tokenizerData.addedTokens.array(or: []) {
            if let index = value.id.integer(), index > maxIndex {
                maxIndex = index
            }
        }
        let vocabSize = max(maxIndex + 1, 1)
        var vocab = Array(repeating: "", count: vocabSize)
        
        for (key, value) in tokenizerData.model.vocab.dictionary(or: [:]) {
            if let index = value.integer(), vocab.indices.contains(index) {
                vocab[index] = key.string
            }
        }
        
        for value in tokenizerData.addedTokens.array(or: []) {
            if let index = value.id.integer(), let token = value.content.string(), vocab.indices.contains(index) {
                vocab[index] = token
            }
        }
        
        let decoders: [Config] = switch tokenizerData.decoder.type.string() {
        case "Sequence":
            tokenizerData.decoder.decoders.array(or: [])
        default:
            [tokenizerData.decoder]
        }
        
        var vocabType: Int32 = 0
        loop: for decoder in decoders {
            switch decoder.type.string() {
            case "ByteFallback":
                vocabType = 1
                break loop
            case "ByteLevel":
                vocabType = 2
                break loop
            default:
                continue
            }
        }
        
        var stopTokenIds: [Int32] = configuration.extraEOSTokens.compactMap(vocab.firstIndex).map(Int32.init)
        if let tokenizerConfig, let eosToken = tokenizerConfig.eosToken.string(), let eosTokenId = vocab.firstIndex(of: eosToken) {
            stopTokenIds.append(Int32(eosTokenId))
        }
        
//        print("Vocab size:", vocab.count)
//        print("Vocab type:", vocabType)
//        print("Stop tokens Ids:", stopTokenIds)
//        print("Grammar:", grammar)
              
        let grammarMatcher = try XGrammar(vocab: vocab, vocabType: vocabType, stopTokenIds: stopTokenIds, grammar: grammar)
        let processor = GrammarMaskedLogitProcessor(grammarMatcher: grammarMatcher)
        return processor
    }
}
