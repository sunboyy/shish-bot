CREATE TABLE `user` (
    `user_id` VARCHAR(63) NOT NULL PRIMARY KEY,
    `mode` VARCHAR(6) NOT NULL,
    `last_eval_id` INT,
    `description` VARCHAR(255)
);

CREATE TABLE `response` (
    `response_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `user_id` VARCHAR(63) NOT NULL,
    `conv_id` INT NOT NULL,
    `answer` VARCHAR(10) NOT NULL,
    FOREIGN KEY (`user_id`) REFERENCES `user`(`user_id`)
);

CREATE TABLE `chat` (
    `chat_id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `user_id` VARCHAR(63) NOT NULL,
    `message` TEXT NOT NULL,
    `response` TEXT NOT NULL,
    FOREIGN KEY (`user_id`) REFERENCES `user`(`user_id`)
);
