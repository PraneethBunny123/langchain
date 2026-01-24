import {ChatOllama} from "@langchain/ollama";
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from "@langchain/core/prompts";

// creating ollama model instance, 
// ChatOllama is specifically designed for chat-based models that wraps the langchain functionality. 
// It runs locally if you have ollama installed and use the existing pulled models.
const model = new ChatOllama({
    model: "llama3",
})

// genreate response from the model
const response = await model.invoke("Hello")

console.log(response.content);

// // creating prompt template (this will help to create dynamic prompts)
// const prompt = ChatPromptTemplate.fromTemplate("Tell me a joke about {input}."); // one way to create prompt template

const prompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate("short and funny joke"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
])

// console.log(await prompt.format({input: "cat"}))

// // create chain (connect prompt template with model)
const chain = prompt.pipe(model)

// // call chain with input
const result = await chain.invoke({input: "Cat"})
console.log(result);
