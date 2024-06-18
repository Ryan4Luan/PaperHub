from setuptools import setup, find_packages

setup(
    name='MedChatbotRAG',
    version='0.1.0',
    author='Jiufeng Li',
    author_email='jiufeng.li@stud.uni-heidelberg.de',
    description='A Medical Question-Answering Demo Based On RAG',
    long_description=open('..\README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Aayushtirmalle/QA-Robot-Med-INLPT-WS2023',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'pandas',
        'tqdm',
        'requests',
        'streamlit',
        'langchain',
        'pinecone-client',
        'openai',
        'st-annotated_text',
        'markdown',
        'tiktoken'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
)
