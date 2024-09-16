# ragas based RAG LLM performance analysis

## Project Overview
This project implements a pipeline for web content extraction, text embedding, and question-answering using **LangChain**, **OpenAI**, and **Ragas** for evaluation. The system scrapes web pages using **Selenium**, processes the content into embeddings, performs queries using a large language model (LLM), and evaluates the quality of the responses using multiple metrics.

## Theoretical Explanation and Breakdown of Functions

### 1. **Loading Environment Variables**
   ```python
   load_dotenv()
   api_key = os.environ.get("OPENAI_API_KEY")
   openai.api_key = api_key
   ```
   - **Purpose**: Loads environment variables from a `.env` file, specifically to retrieve the OpenAI API key. This key is necessary for accessing OpenAI's LLM and embedding services.
   - **Theoretical Context**: Externalizing sensitive information like API keys into environment variables is a common practice for enhancing security and flexibility, allowing the same code to run in multiple environments without hardcoding sensitive data.

### 2. **Web Scraping with Selenium**
   ```python
   def load_content_with_selenium(url):
       chrome_options = Options()
       chrome_options.add_argument("--headless")
       chrome_options.add_argument("--disable-gpu")
       driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
       driver.set_page_load_timeout(10)
       try:
           driver.get(url)
           loader = SeleniumURLLoader(urls=[url])
           documents = loader.load()
           return documents
       finally:
           driver.quit()
   ```
   - **Purpose**: Uses Selenium to scrape web pages. The function configures Chrome to run in **headless mode** (without a GUI), sets up a WebDriver, and fetches the content from the provided URLs.
   - **Theoretical Context**: Web scraping is an essential technique for extracting structured or unstructured data from web pages. Selenium automates this process by controlling a web browser to simulate user interactions. **Headless mode** is used for performance, as it eliminates the need for rendering a visual interface.

### 3. **Processing Documents into Plain Text**
   ```python
   documentList = []
   for doc in documents:
       d = str(doc.page_content).replace("\\n", " ").replace("\\t"," ").replace("\n", " ").replace("\t", " ")
       documentList.append(d)
   ```
   - **Purpose**: Converts the raw content of the web pages into plain text by stripping out special characters like `\n` (newlines) and `\t` (tabs) to make the content easier to process.
   - **Theoretical Context**: Preprocessing of text is a fundamental step in natural language processing (NLP) pipelines. Cleaning and normalizing text ensures that the downstream tasks (like tokenization, embedding, or querying) operate on well-structured data, leading to more accurate results.

### 4. **Text Chunking and Splitting**
   ```python
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
   texts = text_splitter.split_text(''.join(documentList))
   ```
   - **Purpose**: Breaks the document text into smaller, overlapping chunks using the `RecursiveCharacterTextSplitter`. Each chunk is 1000 characters long, with an overlap of 50 characters between chunks.
   - **Theoretical Context**: LLMs have a maximum input size (token limit), so long texts need to be split into manageable chunks. Overlapping chunks are used to preserve context between adjacent text segments. This ensures that when the chunks are processed individually, they retain sufficient contextual information to produce meaningful embeddings and query results.

### 5. **Embedding the Text Using OpenAI**
   ```python
   embeddings = OpenAIEmbeddings(model=embeddings_deployment)
   all_texts.extend(texts)
   ```
   - **Purpose**: Converts each chunk of text into high-dimensional vectors (embeddings) using OpenAI’s embedding model (`text-embedding-3-large`). These embeddings represent the semantic meaning of the text, allowing for efficient similarity-based search.
   - **Theoretical Context**: Text embeddings are a core technique in NLP. By mapping textual content to a vector space, semantically similar pieces of text will have embeddings that are close to each other in that space. This is critical for retrieval tasks, where the goal is to find relevant information based on a query.

### 6. **Creating the Vector Store Using Chroma**
   ```python
   vector_store = Chroma.from_texts(all_texts, embeddings)
   ```
   - **Purpose**: Stores the embeddings in a **Chroma vector store**, which allows for efficient retrieval of relevant document chunks based on user queries.
   - **Theoretical Context**: A vector store is a specialized database that enables fast retrieval of similar vectors (embeddings) through algorithms like **k-nearest neighbors**. This is used extensively in information retrieval systems to quickly find relevant documents based on similarity metrics.

### 7. **Setting Up the Question-Answer Chain**
   ```python
   PROMPT_TEMPLATE = """
   Go through the context and answer given question strictly based on context.
   Context: {context}
   Question: {question}
   Answer:
   """
   qa_chain = RetrievalQA.from_chain_type(
       llm=ChatOpenAI(temperature=0),
       retriever=vector_store.as_retriever(search_kwargs={'k': 1}),
       return_source_documents=True,
       chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
   )
   ```
   - **Purpose**: This sets up a **retrieval-based question-answering chain**. It uses a pre-defined template (`PROMPT_TEMPLATE`) to guide the LLM's responses. The retrieval mechanism searches the vector store for the most relevant document chunks, which are passed to the LLM for generating an answer.
   - **Theoretical Context**: Retrieval-Augmented Generation (RAG) combines information retrieval and language generation to answer questions based on external documents. Instead of relying solely on the LLM’s internal knowledge, relevant external content is retrieved and used as the context for the model’s output. This improves accuracy, especially when the LLM might lack specific knowledge.

### 8. **Running Queries**
   ```python
   queries = [
       "Who discovered the Galapagos Islands and how?",
       "What is the economic significance of New York City?",
       ...
   ]
   for query in queries:
       result = qa_chain.invoke({"query": query})
       results.append(result['result'])
   ```
   - **Purpose**: Queries the LLM using the pre-set question-answer chain. For each query, it retrieves the relevant document chunks and generates an answer, which is appended to the `results` list.
   - **Theoretical Context**: Querying an LLM with context is a key feature of retrieval-augmented systems. By combining relevant external context with the model’s generative capabilities, you get factually grounded answers, which is crucial for tasks like fact-checking or answering specific domain-based questions.

### 9. **Evaluating Results Using Ragas**
   ```python
   dataset = Dataset.from_dict(d)
   score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, ...])
   ```
   - **Purpose**: Evaluates the quality of the LLM’s responses using the **Ragas** framework. Metrics such as **faithfulness**, **relevancy**, **context precision**, and **answer correctness** are calculated to assess the performance of the model.
   - **Theoretical Context**: Evaluation metrics in NLP are crucial for validating the performance of language models. Metrics like **faithfulness** ensure that the model’s answers align with the provided context, while **relevancy** checks whether the answer directly addresses the query. These metrics help quantify the model’s reliability and effectiveness in real-world applications.

### 10. **Exporting Results**
   ```python
   score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
   ```
   - **Purpose**: Saves the evaluation results to a CSV file for further analysis or reporting.
   - **Theoretical Context**: Exporting results to a structured format like CSV is important for documenting and sharing findings. This is particularly useful in research and production environments where results need to be reviewed, compared, or used for reporting purposes.

## Summary
This code implements a sophisticated pipeline for document-based question answering. It starts by scraping content, processing it into embeddings, and setting up a retrieval-based system where the LLM uses external context to generate answers. Evaluation using **Ragas** ensures that the generated answers are measured against quality metrics such as faithfulness and relevancy.
