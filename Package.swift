// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-swift-structured",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [.library(name: "MLXStructured", targets: ["MLXStructured"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.29.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/petrukha-ivan/swift-json-schema", from: "2.0.2"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
    ],
    targets: [
        // C package
        .target(
            name: "CMLXStructured",
            exclude: [
                "xgrammar/web",
                "xgrammar/tests",
                "xgrammar/3rdparty/cpptrace",
                "xgrammar/3rdparty/googletest",
                "xgrammar/3rdparty/dlpack/contrib",
                "xgrammar/3rdparty/picojson",
                "xgrammar/cpp/nanobind",
            ],
            cSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
            ],
            cxxSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
            ]
        ),
        // Main package
        .target(
            name: "MLXStructured",
            dependencies: [
                .target(name: "CMLXStructured"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "JSONSchema", package: "swift-json-schema")
            ]
        ),
        // CLI for testing
        .executableTarget(
            name: "MLXStructuredCLI",
            dependencies: [
                .target(name: "MLXStructured"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
        ),
        // Unit tests
        .testTarget(
            name: "MLXStructuredTests",
            dependencies: [
                .target(name: "MLXStructured"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ],
        ),
    ],
    cxxLanguageStandard: .gnucxx17
)
