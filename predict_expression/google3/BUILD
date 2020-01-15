py_library(
    name = "vikramtrax",
    srcs = ["vikramtrax.py"],
    deps = [
        "//pyglib:gfile",
        "//third_party/py/gin",
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/py/numpy",
        # "//third_party/py/trax:backend",
        "//third_party/py/trax/supervised:inputs",
    ],
)
filegroup(
    name = "gin_files",
    srcs = glob(["configs/*.gin"]),
)
py_binary(
    name = "trainer",
    srcs = ["//third_party/py/trax:trainer.py"],
    data = [":gin_files"],
    paropts = [
        "--compress",
        "--allow_missing_modules",
    ],
    python_version = "PY3",
    deps = [
        ":vikramtrax",
        "//learning/brain/research/conversation/meena/models/t2t/data_generators:meena_problems",
        "//learning/deepmind/xmanager2/client/google",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
        "//third_party/py/trax/models",
        "//third_party/py/trax/optimizers",
        "//third_party/py/trax/supervised:trainer_lib",
    ],
)
