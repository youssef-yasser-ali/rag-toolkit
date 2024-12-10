from setuptools import setup, find_packages

setup(
    name="rag_toolkit",  
    version="0.1.0", 
    packages=find_packages(),
    description="A toolkit for RAG systems",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Youssef Yasser",
    author_email="yyasser849@gmail.com",
    # url="https://github.com/youssefyasserali/rag-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7', 
    install_requires=[
        "langchain>=0.3.7", 
        "langchain-community>=0.3.5", 
        "langchain-core>=0.3.20" 
    ],
)
