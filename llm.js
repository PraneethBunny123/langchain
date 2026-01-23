// import ollama from 'ollama'

// const response = await ollama.chat({
//   model: 'gemma3',
//   messages: [{ role: 'user', content: 'Hello' }],
// })

// console.log(response)

import dotenv from 'dotenv'
dotenv.config()
import { Ollama } from 'ollama'

const ollama = new Ollama({
    host: 'https://ollama.com',
    headers: { Authorization: 'Bearer ' + process.env.OLLAMA_API_KEY },
})

const response = await ollama.chat({
    model: 'gpt-oss:120b',
    messages: [{ role: 'user', content: 'Hello' }],
})

console.log(response);

// for await (const part of response) {
//     process.stdout.write(part.message.content)
// }