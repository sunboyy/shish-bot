const mysql = require('mysql')
const { db } = require('./config')

const pool = mysql.createPool(db)

function getConnection() {
    return new Promise((resolve, reject) => {
        pool.getConnection((err, conn) => {
            if (err) reject(err)
            else resolve(conn)
        })
    })
}

async function getUser(userId) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('SELECT * FROM `user` WHERE `user_id` = ?', [userId], (err, rows) => {
            if (err) reject(err)
            else if (rows.length == 0) resolve(undefined)
            else resolve(rows[0])
            conn.release()
        })
    })
}

async function insertUser(userId) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('INSERT INTO `user` (`user_id`, `mode`) VALUES (?, \'chat\')', [userId], (err, res) => {
            if (err) reject(err)
            else resolve()
            conn.release()
        })
    })
}

async function changeUserMode(userId, mode) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('UPDATE `user` SET `mode` = ? WHERE `user_id` = ?', [mode, userId], (err, res) => {
            if (err) reject(err)
            else resolve()
            conn.release()
        })
    })
}

async function getAnsweredIds(userId) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('SELECT * FROM `response` WHERE `user_id` = ?', [userId], (err, rows) => {
            if (err) reject(err)
            else resolve(rows.map(res => res.conv_id))
            conn.release()
        })
    })
}

async function insertResponse(userId, convId, answer) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('INSERT INTO `response` (`user_id`, `conv_id`, `answer`) VALUES (?, ?, ?)', [userId, convId, answer], (err, res) => {
            if (err) reject(err)
            else resolve()
            conn.release()
        })
    })
}

async function updateLastEvalId(userId, convId) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('UPDATE `user` SET `last_eval_id` = ? WHERE `user_id` = ?', [convId, userId], (err, res) => {
            if (err) reject(err)
            else resolve()
            conn.release()
        })
    })
}

async function insertChat(userId, message, response) {
    const conn = await getConnection()
    return new Promise((resolve, reject) => {
        conn.query('INSERT INTO `chat` (`user_id`, `message`, `response`) VALUES (?, ?, ?)', [userId, message, response], (err, res) => {
            if (err) reject(err)
            else resolve()
            conn.release()
        })
    })
}

module.exports = { getUser, insertUser, changeUserMode, insertResponse, getAnsweredIds, updateLastEvalId, insertChat }
