// import ollama from 'ollama'

// const response = await ollama.chat({
//   model: 'gemma3',
//   messages: [{ role: 'user', content: 'Hello' }],
// })

// console.log(response)

import dotenv from 'dotenv'
dotenv.config()
import { Ollama } from 'ollama'

const model = new Ollama({
    host: 'https://ollama.com',
    headers: { Authorization: 'Bearer ' + process.env.OLLAMA_API_KEY },
})

const response = await model.generate({
    model: 'gpt-oss:120b',
    prompt: 'Hello',
})

console.log(response);
