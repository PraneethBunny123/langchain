import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

// load data and create vector store
async function createVectorStore() {
    const loader = new CheerioWebBaseLoader("https://www.blog.langchain.com/langchain-expression-language/")
    const docs = await loader.load()


    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 200, chunkOverlap: 20})
    const splitDocs = await splitter.splitDocuments(docs)

    const embeddings = new OllamaEmbeddings()
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings)

    return vectorStore
}

// create retrieval chain
async function createChain(vectorStore) {

    const model = new ChatOllama({
        model: "llama3"
    })

    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system", 
            "Answer the user's questions based on the following context: {context}"
        ],
        new MessagesPlaceholder("chat_history"),
        [
            "human", 
            "{input}"
        ],
    ])

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt
    })

    const retriever = vectorStore.asRetriever({k: 2})

    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever
    })

    return conversationChain
}

const vectorStore = await createVectorStore()
const chain = await createChain(vectorStore)

// fake chat history for testing
const chatHistory = [
    new HumanMessage("Hello"),
    new AIMessage("Hi, How can I help you?"),
    new HumanMessage("My name is debby"),
    new AIMessage("Hi Debby, How can I help you?"),
    new HumanMessage("What is LECL?"),
    new AIMessage("LECL stands for Langchain Expression Language"),
]

const response = await chain.invoke({ 
    input: "what is my name?",
    chat_history: chatHistory
})

console.log(response);
