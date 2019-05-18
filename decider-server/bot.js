const axios = require('axios')
const { botUrl } = require('./config')

async function getReply(q) {
    const response = await axios.default.post(`${botUrl}/chat`, { q })
    return response.data
}

module.exports = { getReply }
