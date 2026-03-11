import CMLXStructured
import Foundation

private enum BenchmarkError: Error {
    case nativeError(String)
    case initializationFailed(String)
}

private final class ErrorCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var message: String?

    func set(_ message: String?) {
        lock.withLock { self.message = message }
    }

    func take() -> String? {
        lock.withLock {
            defer { message = nil }
            return message
        }
    }
}

private let errorCapture = ErrorCapture()

private let errorHandler: @convention(c) (UnsafePointer<CChar>?) -> Void = { pointer in
    errorCapture.set(pointer.map { String(cString: $0) })
}

private struct DLDevice {
    var deviceType: Int32
    var deviceId: Int32
}

private struct DLDataType {
    var code: UInt8
    var bits: UInt8
    var lanes: UInt16
}

private struct DLTensor {
    var data: UnsafeMutableRawPointer?
    var device: DLDevice
    var ndim: Int32
    var dtype: DLDataType
    var shape: UnsafeMutablePointer<Int64>?
    var strides: UnsafeMutablePointer<Int64>?
    var byteOffset: UInt64

    static func nextTokenBitmask(bufferSize: Int) -> DLTensor {
        let dataBytes = bufferSize * MemoryLayout<Int32>.stride
        let data = UnsafeMutableRawPointer.allocate(byteCount: dataBytes, alignment: 64)
        data.bindMemory(to: Int32.self, capacity: bufferSize).initialize(repeating: 0, count: bufferSize)

        let shape = UnsafeMutablePointer<Int64>.allocate(capacity: 1)
        shape.initialize(to: Int64(bufferSize))

        return DLTensor(
            data: data,
            device: DLDevice(deviceType: 1, deviceId: 0),
            ndim: 1,
            dtype: DLDataType(code: 0, bits: 32, lanes: 1),
            shape: shape,
            strides: nil,
            byteOffset: 0
        )
    }

    mutating func deallocate() {
        data?.deallocate()
        shape?.deallocate()
        strides?.deallocate()
        data = nil
        shape = nil
        strides = nil
    }
}

private struct NativeMatcher {
    let vocabSize: Int
    let bufferSize: Int
    let matcher: UnsafeMutableRawPointer
    private let compiledGrammar: UnsafeMutableRawPointer
    private let tokenizerInfo: UnsafeMutableRawPointer
    private let duplicatedVocab: [UnsafeMutablePointer<CChar>?]

    init(vocab: [String], stopTokenIds: [Int32], ebnfGrammar: String) throws {
        let duplicatedVocab = vocab.map { strdup($0) }
        self.duplicatedVocab = duplicatedVocab
        self.vocabSize = vocab.count
        self.bufferSize = (vocab.count + 31) / 32

        let tokenizerInfo = duplicatedVocab.map { pointer in
            pointer.map { UnsafePointer($0) }
        }.withUnsafeBufferPointer { vocabBuffer in
            stopTokenIds.withUnsafeBufferPointer { eosBuffer in
                tokenizer_info_new(
                    vocabBuffer.baseAddress,
                    vocabBuffer.count,
                    0,
                    eosBuffer.baseAddress,
                    eosBuffer.count
                )
            }
        }
        guard let tokenizerInfo else {
            Self.cleanup(vocab: duplicatedVocab, tokenizerInfo: nil, compiledGrammar: nil, matcher: nil)
            throw BenchmarkError.initializationFailed(errorCapture.take() ?? "tokenizer_info_new failed")
        }

        let compiledGrammar = ebnfGrammar.utf8CString.withUnsafeBufferPointer {
            compile_ebnf_grammar(tokenizerInfo, $0.baseAddress, $0.count)
        }
        guard let compiledGrammar else {
            Self.cleanup(vocab: duplicatedVocab, tokenizerInfo: tokenizerInfo, compiledGrammar: nil, matcher: nil)
            throw BenchmarkError.initializationFailed(errorCapture.take() ?? "compile_ebnf_grammar failed")
        }

        guard let matcher = grammar_matcher_new(compiledGrammar) else {
            Self.cleanup(vocab: duplicatedVocab, tokenizerInfo: tokenizerInfo, compiledGrammar: compiledGrammar, matcher: nil)
            throw BenchmarkError.initializationFailed(errorCapture.take() ?? "grammar_matcher_new failed")
        }

        self.tokenizerInfo = tokenizerInfo
        self.compiledGrammar = compiledGrammar
        self.matcher = matcher
    }

    func reset() {
        grammar_matcher_reset(matcher)
    }

    func tearDown() {
        Self.cleanup(
            vocab: duplicatedVocab,
            tokenizerInfo: tokenizerInfo,
            compiledGrammar: compiledGrammar,
            matcher: matcher
        )
    }

    private static func cleanup(
        vocab: [UnsafeMutablePointer<CChar>?],
        tokenizerInfo: UnsafeMutableRawPointer?,
        compiledGrammar: UnsafeMutableRawPointer?,
        matcher: UnsafeMutableRawPointer?
    ) {
        if let matcher {
            grammar_matcher_free(matcher)
        }
        if let compiledGrammar {
            compiled_grammar_free(compiledGrammar)
        }
        if let tokenizerInfo {
            tokenizer_info_free(tokenizerInfo)
        }
        for pointer in vocab {
            free(pointer)
        }
    }
}

private struct BenchmarkResult {
    let label: String
    let totalSeconds: Double
    let iterations: Int

    var microsecondsPerIteration: Double {
        (totalSeconds / Double(iterations)) * 1_000_000
    }
}

private func benchmark(label: String, iterations: Int, _ body: () throws -> Void) rethrows -> BenchmarkResult {
    let clock = ContinuousClock()
    let start = clock.now
    for _ in 0..<iterations {
        try body()
    }
    let elapsed = start.duration(to: clock.now)
    return BenchmarkResult(
        label: label,
        totalSeconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18,
        iterations: iterations
    )
}

private func oldSwiftStyleDenseMask(bitmaskTensor: inout DLTensor, vocabSize: Int) -> [Float] {
    let buffer = UnsafeBufferPointer(
        start: bitmaskTensor.data?.assumingMemoryBound(to: Int32.self),
        count: (vocabSize + 31) / 32
    )
    var mask = [Float](repeating: -.infinity, count: vocabSize)
    for tokenID in 0..<vocabSize {
        let block = tokenID >> 5
        let bit = tokenID & 31
        if (UInt32(bitPattern: buffer[block]) & (UInt32(1) << UInt32(bit))) != 0 {
            mask[tokenID] = 0
        }
    }
    return mask
}

private func percentile(_ sortedValues: [Double], _ p: Double) -> Double {
    guard !sortedValues.isEmpty else { return 0 }
    let index = Int((Double(sortedValues.count - 1) * p).rounded())
    return sortedValues[index]
}

private func describe(results: [Double], label: String) {
    let sorted = results.sorted()
    let average = sorted.reduce(0, +) / Double(sorted.count)
    print(
        "\(label): avg \(String(format: "%.2f", average)) us, " +
        "p50 \(String(format: "%.2f", percentile(sorted, 0.50))) us, " +
        "p95 \(String(format: "%.2f", percentile(sorted, 0.95))) us"
    )
}

@MainActor
private func run() throws {
    set_error_handler(errorHandler)

    let vocab = ["<eos>"] + (0...0x0FFF).compactMap { UnicodeScalar($0).map(String.init) }
    let payload = "structured-output"
    let ebnfGrammar = #"root ::= ""# + payload + #"""#
    let matcher = try NativeMatcher(vocab: vocab, stopTokenIds: [0], ebnfGrammar: ebnfGrammar)
    defer { matcher.tearDown() }

    let acceptedSequence = payload.map(String.init).compactMap { vocab.firstIndex(of: $0) }.map(Int32.init) + [0]
    var bitmask = DLTensor.nextTokenBitmask(bufferSize: matcher.bufferSize)
    defer { bitmask.deallocate() }
    var denseMask = [Float](repeating: -.infinity, count: matcher.vocabSize)

    let warmupIterations = 50
    for _ in 0..<warmupIterations {
        matcher.reset()
        for token in acceptedSequence.dropLast() {
            guard withUnsafeMutablePointer(to: &bitmask, { grammar_matcher_fill_next_token_bitmask(matcher.matcher, $0) }) else {
                throw BenchmarkError.nativeError(errorCapture.take() ?? "fill_next_token_bitmask failed during warmup")
            }
            _ = oldSwiftStyleDenseMask(bitmaskTensor: &bitmask, vocabSize: matcher.vocabSize)
            _ = grammar_matcher_accept_token_status(matcher.matcher, token)
        }
    }

    let iterations = 200
    var oldMaskSamples = [Double]()
    var newMaskSamples = [Double]()
    var oldAcceptSamples = [Double]()
    var newAcceptSamples = [Double]()
    oldMaskSamples.reserveCapacity(iterations)
    newMaskSamples.reserveCapacity(iterations)
    oldAcceptSamples.reserveCapacity(iterations)
    newAcceptSamples.reserveCapacity(iterations)

    let oldMaskAggregate = try benchmark(label: "old-mask", iterations: iterations) {
        matcher.reset()
        for token in acceptedSequence.dropLast() {
            let clock = ContinuousClock()
            let start = clock.now
            guard withUnsafeMutablePointer(to: &bitmask, { grammar_matcher_fill_next_token_bitmask(matcher.matcher, $0) }) else {
                throw BenchmarkError.nativeError(errorCapture.take() ?? "fill_next_token_bitmask failed")
            }
            _ = oldSwiftStyleDenseMask(bitmaskTensor: &bitmask, vocabSize: matcher.vocabSize)
            let elapsed = start.duration(to: clock.now)
            oldMaskSamples.append((Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18) * 1_000_000)

            let acceptStart = clock.now
            if grammar_matcher_is_terminated(matcher.matcher) {
                break
            }
            let accepted = grammar_matcher_accept_token(matcher.matcher, token)
            let acceptElapsed = acceptStart.duration(to: clock.now)
            oldAcceptSamples.append((Double(acceptElapsed.components.seconds) + Double(acceptElapsed.components.attoseconds) / 1e18) * 1_000_000)
            if !accepted {
                throw BenchmarkError.nativeError("old accept path rejected token \(token)")
            }
        }
    }

    let newMaskAggregate = try benchmark(label: "new-mask", iterations: iterations) {
        matcher.reset()
        for token in acceptedSequence.dropLast() {
            let clock = ContinuousClock()
            let start = clock.now
            let filled = withUnsafeMutablePointer(to: &bitmask) { bitmaskPointer in
                denseMask.withUnsafeMutableBufferPointer { denseMaskPointer in
                    grammar_matcher_fill_next_token_dense_mask(
                        matcher.matcher,
                        bitmaskPointer,
                        denseMaskPointer.baseAddress,
                        denseMaskPointer.count
                    )
                }
            }
            let elapsed = start.duration(to: clock.now)
            newMaskSamples.append((Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18) * 1_000_000)
            guard filled else {
                throw BenchmarkError.nativeError(errorCapture.take() ?? "fill_next_token_dense_mask failed")
            }

            let acceptStart = clock.now
            let status = grammar_matcher_accept_token_status(matcher.matcher, token)
            let acceptElapsed = acceptStart.duration(to: clock.now)
            newAcceptSamples.append((Double(acceptElapsed.components.seconds) + Double(acceptElapsed.components.attoseconds) / 1e18) * 1_000_000)
            if status == 0 {
                throw BenchmarkError.nativeError("new accept path rejected token \(token)")
            }
        }
    }

    print("Structured generation matcher microbenchmark")
    print("Vocab size: \(matcher.vocabSize)")
    print("Sequence length: \(acceptedSequence.count)")
    print("Iterations: \(iterations)")
    print("")

    print("Mask path")
    print(
        "old total: \(String(format: "%.4f", oldMaskAggregate.totalSeconds)) s, " +
        "\(String(format: "%.2f", oldMaskAggregate.microsecondsPerIteration)) us/iteration"
    )
    print(
        "new total: \(String(format: "%.4f", newMaskAggregate.totalSeconds)) s, " +
        "\(String(format: "%.2f", newMaskAggregate.microsecondsPerIteration)) us/iteration"
    )
    let maskSpeedup = oldMaskAggregate.totalSeconds / newMaskAggregate.totalSeconds
    let maskReduction = (1 - (newMaskAggregate.totalSeconds / oldMaskAggregate.totalSeconds)) * 100
    print(
        "mask speedup: \(String(format: "%.2f", maskSpeedup))x " +
        "(\(String(format: "%.1f", maskReduction))% less time)"
    )
    describe(results: oldMaskSamples, label: "old mask samples")
    describe(results: newMaskSamples, label: "new mask samples")
    print("")

    let oldAcceptAverage = oldAcceptSamples.reduce(0, +) / Double(oldAcceptSamples.count)
    let newAcceptAverage = newAcceptSamples.reduce(0, +) / Double(newAcceptSamples.count)
    let acceptSpeedup = oldAcceptAverage / newAcceptAverage
    let acceptReduction = (1 - (newAcceptAverage / oldAcceptAverage)) * 100
    print("Accept path")
    print("old avg: \(String(format: "%.2f", oldAcceptAverage)) us/call")
    print("new avg: \(String(format: "%.2f", newAcceptAverage)) us/call")
    print(
        "accept speedup: \(String(format: "%.2f", acceptSpeedup))x " +
        "(\(String(format: "%.1f", acceptReduction))% less time)"
    )
    describe(results: oldAcceptSamples, label: "old accept samples")
    describe(results: newAcceptSamples, label: "new accept samples")
}

Task { @MainActor in
    do {
        try run()
    } catch {
        fputs("Benchmark failed: \(error)\n", stderr)
        exit(1)
    }
    exit(0)
}
dispatchMain()
