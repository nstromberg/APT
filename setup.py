from distutils.core import setup

ext_modules = []
cmdclass = {}

setup(
    name="apt",
    version="1.0.0",
    description="",
    url="https://github.com/yulun-rayn/adversarial-zero-shot-prediction",
    author="Yulun Wu",
    author_email="yulun_wu@berkeley.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
