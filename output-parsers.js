import {ChatOllama} from "@langchain/ollama";
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from "@langchain/core/prompts";
import { CommaSeparatedListOutputParser, StringOutputParser, StructuredOutputParser } from "@langchain/core/output_parsers";
import {z} from "zod";

const model = new ChatOllama({
    model: "llama3",
})

async function callStringOutputParser() {
    const prompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate("Short and funny joke based on the input provide by the user"),
        HumanMessagePromptTemplate.fromTemplate("{input}"),
    ])

    const stringParser = new StringOutputParser();
    const chain = prompt.pipe(model).pipe(stringParser)

    const result = await chain.invoke({input: "Cat"})
    return result;
}

async function callListOutputParser() {
    const prompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate("Provide 5 synonyms in a comma separated list for the input provided by the user"),
        HumanMessagePromptTemplate.fromTemplate("{input}"),
    ])

    const listParser = new CommaSeparatedListOutputParser()
    const chain = prompt.pipe(model).pipe(listParser)

    const result = await chain.invoke({input: "heed"})
    return result;
}

async function callStructuredOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        extract infomation from the following text. 
        Formatting Instructions: {format_instructions}
        text: {text}
    `)

    const structuredParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "the name of the person",
        age: "the age of the person"
    })

    const chain = prompt.pipe(model).pipe(structuredParser)
    const result = await chain.invoke({
        text: "Max is 30 years old",
        format_instructions: structuredParser.getFormatInstructions()
    })

    return result
}

async function callZodOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        extract infomation from the following text. 
        Formatting Instructions: {format_instructions}
        text: {text}
    `)

    const zodParser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("name of the recipe"),
            ingredients: z.array(z.string()).describe("All ingredients to the receipe")
        })
    )

    const chain = prompt.pipe(model).pipe(zodParser)
    const result = await chain.invoke({
        text: "Can you tell me the recipe of shrimp Scampi with ingredients?",
        format_instructions: zodParser.getFormatInstructions()
    })

    return result
}

const response = await callZodOutputParser()
console.log(response);