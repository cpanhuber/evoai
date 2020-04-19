cc_library(
    name = "evoai_lib",
    hdrs = glob(
        [
            "include/evoai/**/*.h",
        ],
    ),
    srcs = glob(
        [
            "src/**/*.cpp",
        ],
    ),
    deps = [
        "@eigen",
    ],
    copts = ["-Wall", "-Werror", "--pedantic"],
    strip_include_prefix = "include/",
)

cc_binary(
    name = "xor",
    srcs = ["src/examples/xor.cpp"],
    deps = [":evoai_lib"],
    copts = ["-Wall", "-Werror", "--pedantic"],
)

cc_test(
    name = "evoai_spec",
    srcs = glob(["test/*.cpp", "test/**/*.h"]),
    deps = [
        "@gtest//:gtest_main",
        ":evoai_lib",
    ],
    copts = ["-Wall", "-Werror", "--pedantic"],
)
