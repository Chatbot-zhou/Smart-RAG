"""
需求：实现"基于mysql数据库构建问答系统"中mysql相关的操作，包括: 加载高频FAQ数据到mysql、拉取所有的问题、通过问题检索答案 功能
思路步骤：
1. 实现mysql连接初始化，拿到client对象
    1.1 基于配置文件，在构造方法中连接mysql，并拿到cursor对象
    1.2 实现close方法，用于断开连接
2. 实现高频FAQ数据加载mysql功能
    2.1 实现FAQ表建表操作
    2.2 实现读取csv文件并插入mysql的功能
3. 实现拉取所有问题的方法
4. 实现通过问题检索对应答案的方法
"""

import pymysql
from base.config import single_config
from base.logger import single_logger
from pymysql import MySQLError

import pandas

class MysqlClient(object):
    # 1. 实现mysql连接初始化，拿到client对象
    def __init__(self):
        # 1.1 基于配置文件，在构造方法中连接mysql，并拿到cursor对象
        self.host = single_config.MYSQL_HOST
        self.user = single_config.MYSQL_USER
        self.password = single_config.MYSQL_PASSWORD
        self.db = single_config.MYSQL_DATABASE
        self.logger = single_logger
        try:
            # 连接：用来控制数据库连接
            self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.db)
            # 游标：用来执行SQL，操作具体的SQL执行的
            self.cursor = self.connection.cursor()
            self.logger.info('mysql连接成功！')
        except MySQLError as e:
            self.logger.error(f'mysql连接失败：{e}')
            raise

    # 1.2 实现close方法，用于断开连接
    def close(self):
        self.cursor.close()
        self.connection.close()

    # 2. 实现高频FAQ数据加载mysql功能
    # 2.1 实现FAQ表建表操作
    def create_table(self):
        sql = """
              CREATE TABLE IF NOT EXISTS jpkb
              (
                  id
                  INT
                  AUTO_INCREMENT
                  PRIMARY
                  KEY,
                  subject_name
                  VARCHAR
              (
                  20
              ),
                  question VARCHAR
              (
                  1000
              ),
                  answer VARCHAR
              (
                  1000
              )) \
              """
        self.logger.info(f'开始执行SQL：{sql}')
        try:
            self.cursor.execute(sql)
            self.logger.info(f'执行SQL完成')
        except MySQLError as e:
            self.logger.error(f'执行SQL失败{e}')

    # 2.2 实现读取csv文件并插入mysql的功能
    def insert_data(self, csv_path):
        data_frame = pandas.read_csv(csv_path)
        try:
            # 返回两个字段：index , row
            for _, row in data_frame.iterrows():
                # 这种方式是为了安全，防止SQL注入
                sql = "insert into jpkb (subject_name, question, answer) values ( %s, %s, %s)"
                self.cursor.execute(sql, (row['学科名称'], row['问题'], row['答案']))
            # 把所有的数据的插入作为一次事务，要么全部成功
            self.connection.commit()
            self.logger.info(f'数据插入成功，总计：{data_frame.count()} 条')
        except MySQLError as e:
            # 把所有的数据的插入作为一次事务，要么回滚
            self.connection.rollback()
            self.logger.error(f'数据插入失败：{e}')

    # 3. 实现拉取所有问题的方法
    def fetch_questions(self):
        sql = "select question from jpkb"
        try:
            # results：二维表格
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.logger.info(f'获取数据成功，共：{len(results)}条')
            return results
        except MySQLError as e:
            self.logger.error(f'获取数据失败:{e}')
            return []

    # 4. 实现通过问题检索对应答案的方法
    def fetch_answer(self, question):
        sql = "select answer from jpkb where question = %s"
        try:
            # results：二维表格
            self.cursor.execute(sql, (question,))
            # 一维数组，每个元素代表每个字段
            answer = self.cursor.fetchone()
            if answer:
                # 1. 找到了答案
                self.logger.info(f'获取答案成功')
                return answer[0]
            # 2. 没找到答案
            self.logger.warn(f'获取答案失败，问题: {question}。请检查redis和mysql中数据的一致性')
            return None
        except MySQLError as e:
            # 3. 找了，执行失败了
            self.logger.error(f'获取答案执行失败:{e}')
            return None


if __name__ == '__main__':
    mysql_client = MysqlClient()
    mysql_client.create_table()
    mysql_client.insert_data(single_config.FAQ_CSV_PATH)
