import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

const model = new ChatOllama({
    model: "llama3"
})

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "you are a helpful assistant called Ronald"],
    ["human", "{input}"]
])

const chain = await model.invoke("hello")
console.log(chain);
