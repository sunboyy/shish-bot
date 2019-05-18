const express = require('express')
const bodyParser = require('body-parser')
const bot = require('./bot')
const lineSender = require('./line-sender')
const app = express()
const port = process.env.PORT || 4000

let data
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())
app.get('/status', (req, res) => {
    res.json(data)
})
app.post('/webhook/line', async (req, res) => {
    data = req.body
    if (!req.body.events) return res.sendStatus(200)
    let events = req.body.events
    for (let event of events) {
        if (event.message.type != 'text') continue
        let replyToken = event.replyToken
        let msg = event.message.text
        let userId = event.source.userId
        try {
            const reply = await bot.getReply(userId, msg)
            await lineSender.reply(replyToken, reply)
        } catch (e) {
            console.log('Cannot connect to decider server')
            await lineSender.reply(replyToken, "Decider server is not available at the moment.")
        }
    }
    res.sendStatus(200)
})

app.listen(port, () => {
    console.log('Server started')
})
