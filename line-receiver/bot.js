const axios = require('axios')
const { deciderUrl } = require('./config')

async function getReply(userId, message) {
    const response = await axios.default.post(`${deciderUrl}/chat`, { userId, message })
    console.log('Bot: ' + response.status);
    return response.data
}

module.exports = { getReply }
