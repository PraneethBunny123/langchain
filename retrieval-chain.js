import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const model = new ChatOllama({
    model: "llama3"
})

const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the user's question.
        Context: {context}
        Question: {input}
    `)

const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
})

const loader = new CheerioWebBaseLoader("https://www.blog.langchain.com/langchain-expression-language/")
const docs = await loader.load()
// console.log(docs);


const splitter = new RecursiveCharacterTextSplitter({chunkSize: 200, chunkOverlap: 20})
const splitDocs = await splitter.splitDocuments(docs)
console.log(splitDocs)

const response = await chain.invoke({
    input: "what is LCEL?",
    context: docs
})
// console.log(response);
