const express = require('express')
const bodyParser = require('body-parser')
const bot = require('./bot')
const database = require('./database')
const evalData = require('./eval-data.json')
const { port } = require('./config')

const app = express()
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())
app.post('/chat', async (req, res) => {
    let { message, userId } = req.body
    message = message.trim().toLowerCase()
    let user = await database.getUser(userId)
    if (user.description) {
        console.log(user.description + ': ' + message)
    } else {
        console.log(userId + ': ' + message)
    }
    if (!user) {
        await database.insertUser(userId)
        user = await database.getUser(userId)
    }
    if (user.mode === 'chat') {
        if (message === 'eval') {
            await database.changeUserMode(userId, 'eval')
            const answeredIds = await database.getAnsweredIds(userId)
            const remainingEvalData = evalData.filter(d => answeredIds.find(a => a === d.id) === undefined)
            if (remainingEvalData.length === 0) {
                res.send('All data is evaluated')
            } else {
                const randomData = remainingEvalData[Math.floor(Math.random() * remainingEvalData.length)]
                await database.updateLastEvalId(userId, randomData.id)
                res.send('Starting evaluate mode.\nType "human" or "bot" to evaluate\nType "skip" to skip\nType "end" to return to chat mode\n\nQ: ' + randomData.question + '\nA: ' + randomData.answer)
            }
        } else {
            try {
                const response = await bot.getReply(message)
                await database.insertChat(userId, message, response)
                res.send(response)
            } catch (e) {
                console.log('Cannot connect to bot server')
                res.send('Bot server is not available at the moment')
            }
        }
    } else if (user.mode === 'eval') {
        if (message === 'end') {
            await database.changeUserMode(userId, 'chat')
            res.send('End evaluation. Return to chat mode')
        } else if (message !== 'human' && message !== 'bot' && message !== 'skip') {
            res.send('Unknown command. Known command is "human", "bot", "skip" and "end".')
        } else {
            if (message !== 'skip') {
                await database.insertResponse(userId, user.last_eval_id, message)
            }
            const answeredIds = await database.getAnsweredIds(userId)
            const remainingEvalData = evalData.filter(d => answeredIds.find(a => a.id === d.id) === undefined)
            if (remainingEvalData.length === 0) {
                res.send('All data is evaluated')
            } else {
                const randomData = remainingEvalData[Math.floor(Math.random() * remainingEvalData.length)]
                await database.updateLastEvalId(userId, randomData.id)
                res.send('Q: ' + randomData.question + '\nA: ' + randomData.answer)
            }
        }
    }
})

app.listen(port, () => {
    console.log('Decider server started at port ' + port)
})
