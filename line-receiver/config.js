const dotenv = require('dotenv')

dotenv.config()
module.exports = {
    accessToken: '' || process.env.ACCESS_TOKEN,
    deciderUrl: 'http://localhost:3000' || process.env.DECIDER_URL
}
