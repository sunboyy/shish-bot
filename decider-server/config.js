const dotenv = require('dotenv')

dotenv.config()
module.exports = {
    port: 3000 || process.env.PORT,
    db: {
        host: 'localhost' || process.env.MYSQL_HOST,
        user: 'root' || process.env.MYSQL_USER,
        password: '' || process.env.MYSQL_PASSWORD,
        database: 'shishbot' || process.env.MYSQL_DATABASE
    },
    botUrl: 'http://localhost:5000' || process.env.BOT_URL
}
