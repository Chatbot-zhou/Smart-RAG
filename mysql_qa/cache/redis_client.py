"""
需求：实现redis库的读写操作
思路步骤：
1. 初始化redis连接对象
2. 实现写入数据通用方法
3. 实现获取数据通用方法
4. 实现获取答案的方法
5. 实现设置答案的方法
"""
import redis
import json
from redis import RedisError

from base import logger, config


class RedisClient:
    # 1. 初始化redis连接对象
    def __init__(self):
        self.logger = logger.single_logger
        self.config = config.single_config
        try:
            self.client = redis.StrictRedis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                password=self.config.REDIS_PASSWORD,
                db=self.config.REDIS_DB,
                decode_responses=True
            )
        except RedisError as e:
            self.logger.error(f'redis初始化报错：{e}')
            raise e

    # 2. 实现写入数据通用方法
    def set_data(self, key, value):
        # 序列化
        json_str = json.dumps(value)
        try:
            self.client.set(key, json_str)
            self.logger.debug(f'redis写入JSON数据成功:key:{key},value:{json_str}')
        except RedisError as e:
            self.logger.error(f'redis执行失败:{e}')

    # 3. 实现获取数据通用方法
    def get_data(self, key):
        try:
            value_str = self.client.get(key)
            if not value_str:
                self.logger.error(f'redis获取问题数据失败')
                return []
            value = json.loads(value_str)
            self.logger.debug(f'redis读取JSON数据成功:key:{key},value_str:{value_str}')
            return value
        except RedisError as e:
            self.logger.error(f'redis执行失败:{e}')
            return []

    # 4. 实现设置答案的方法
    def set_answer(self, question, answer):
        key = f'answer:{question}'
        try:
            self.client.set(key, answer, ex=60 * 60 * 24)
            self.logger.debug(f'redis写入问答对数据成功:key:{key},answer:{answer}')
        except RedisError as e:
            self.logger.error(f'redis写入问答对数据失败:{e}')

    # 5. 实现获取答案的方法
    def get_answer(self, question):
        key = f'answer:{question}'
        try:
            answer = self.client.get(key)
            if not answer:
                self.logger.info(f'redis没有获取到指定的问答对')
                return ''
            self.logger.debug(f'redis获取数据成功:key:{key},answer:{answer}')
            self.client.expire(key, 60 * 60 * 24)
            return answer
        except RedisError as e:
            self.logger.error(f'redis获取数据失败:{e}')
            return ''
