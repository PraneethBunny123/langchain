import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";

import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { RunnableSequence } from "@langchain/core/runnables";

// load data and create vector store
async function createVectorStore() {
    const loader = new CheerioWebBaseLoader("https://www.blog.langchain.com/langchain-expression-language/")
    const docs = await loader.load()


    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 200, chunkOverlap: 20})
    const splitDocs = await splitter.splitDocuments(docs)

    const embeddings = new OllamaEmbeddings({model: "mxbai-embed-large"})
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
            "ai", 
            "Answer the user's questions based on the following context: {context}"
        ],
        new MessagesPlaceholder("chat_history"),
        [
            "human", 
            "{input}"
        ],
    ])

    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"],
        [
            "human", 
            "Given the above conversation, generate a search query to look up in order to get information relevant to the current question. " +
            "Don't add any preamble, just respond with the query string."
        ],
    ])

    const retriever = vectorStore.asRetriever()

    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt,
    })

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt
    })

    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: RunnableSequence.from([
            (input) => {
                // console.log("original input: ", input);
                return {input}
            }, 
            historyAwareRetriever,
            (docs) => {
                // console.log("Retrieved documents (after rephrasing): ", docs);
                return docs;
            }
        ])
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
    input: "Explain about it?",
    chat_history: chatHistory
})

console.log(response);
