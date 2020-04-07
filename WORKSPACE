workspace(name = "evoai")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gtest",
    urls = ["https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip"],  # 2019-01-07
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
)

# Eigen
http_archive(
    name = "eigen",
    build_file = "//:eigen.BUILD",
    url = "https://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz",
    strip_prefix = "eigen-eigen-323c052e1731",
)
