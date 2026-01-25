import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {Document} from "@langchain/core/documents"
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

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

// Documents
const documents = [
    new Document({
        pageContent: "Today we’re excited to announce a new way of constructing chains. We’re calling this the LangChain Expression Language (in the same spirit as SQLAlchemyExpressionLanguage). This is a declarative way to truly compose chains - and get streaming, batch, and async support out of the box. You can use all the same existing LangChain constructs to create them."
    }),
    new Document({
        pageContent: "The passphrase is LANGCHAIN IS AWESOME"
    })
]

const response = await chain.invoke({
    input: "what is the passphrase?",
    context: documents
})
console.log(response);
