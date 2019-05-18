const axios = require('axios')
const { accessToken } = require('./config')

const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`
}

async function reply(replyToken, msg) {
    const body = JSON.stringify({
        replyToken: replyToken,
        messages: [{
            type: 'text',
            text: msg
        }]
    })
    const response = await axios.post('https://api.line.me/v2/bot/message/reply', body, { headers })
    console.log('Reply: ' + response.status)
}

module.exports = { reply }
