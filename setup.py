from setuptools import setup, find_packages

setup(
    name="rag-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="yyasser849@gmail.com",
    description="A library for building Retrieval-Augmented Generation pipelines.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/youssef-yasser-ali/rag-toolkit",
    packages=find_packages(),
    install_requires=[
        "langchain==0.3.12",
        "langchain-community==0.3.12",
        "langsmith==0.1.140",
        "chromadb==0.5.20",
        "langchain-google-genai==2.0.4",
        "pydantic==2.9.2",
        "pypdf==5.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
