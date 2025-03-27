# RAG-Enabled File Query System

The **RAG-Enabled File Query System** is a tool that allows users to upload files (CSV, JSON, Excel), index their content into a ChromaDB vector database, and query the data using a **Retrieval-Augmented Generation (RAG)** approach. The system retrieves relevant documents from the indexed data and queries an **AI model hosted on LM Studio** to generate intelligent responses based on the provided context.

## Features

- **File Upload**: Supports CSV, JSON, and Excel file uploads.
- **File Indexing**: Indexes the contents of uploaded files into ChromaDB for efficient document retrieval.
- **Data Querying**: Allows users to query the indexed data with natural language questions.
- **AI Integration**: Uses the **LM Studio API** to generate answers based on the context of the retrieved data.
- **Database Management**: Stores file metadata in a SQLite database to track uploaded files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-Enabled-File-Query-System.git
