"""
需求：基于bm25算法实现问题的检索模块，包括：问题加载、分词、BM25 评分、Softmax 归一化，并记录操作日志 等。

因需求比较复杂，这里拆成了两个部分：
1. 数据准备部分：基于mysql、redis中的数据，以及BM25算法类，完成数据加载和算法的初始化
2. 问题检索部分：接收query，计算BM25分值，并返回给用户匹配到的答案
"""
from rank_bm25 import BM25Okapi

"""
需求：基于mysql、redis中的数据，以及BM25算法类，完成数据加载和算法的初始化
实现：对应__init__和_load_data()方法
思路步骤：
1. 初始化bm25检索器类
    1.1 定义算法类中可能用到的对象： 日志、redis、mysql、bm25算法对象、分词问题列表、原始问题列表
2. 加载问题数据
    2.1 尝试从redis中获取：分词问题列表、原始问题列表
    2.2 如果可以获取到，则直接放到内存里 
    2.3 如果redis获取不到，则通过查询mysql的原始问题并进行以下处理： 原始问题列表 -> 分词 -> 分词问题列表
    2.4 初始化bm25模型，并传入：分词问题列表
"""

from base.logger import single_logger
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient
import numpy as np


class BM25Search:
    # 1.1 定义算法类中可能用到的对象： 日志、redis、mysql、bm25算法对象、分词问题列表、原始问题列表
    def __init__(self
                 , mysql_client
                 , redis_client):
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.logger = single_logger

        self.questions = None
        self.tokenizer_questions = None
        self.bm25 = None

        self._loader()

    def _loader(self):
        # 2.1 尝试从redis中获取：分词问题列表、原始问题列表
        question_key = 'qa_origin_question_key'
        tokenized_question_key = 'qa_tokenized_question_key'

        questions = self.redis_client.get_data(question_key)
        tokenizer_questions = self.redis_client.get_data(tokenized_question_key)

        # 2.2 如果可以获取到，则直接放到内存里
        if questions and tokenizer_questions:
            self.questions = questions
            self.tokenizer_questions = tokenizer_questions
        else:
            # 2.3 如果redis获取不到，则通过查询mysql的原始问题并进行以下处理： 原始问题列表 -> 分词 -> 分词问题列表
            # 从mysql获取数据
            all_questions = self.mysql_client.fetch_questions()
            if not all_questions:
                self.logger.error('数据库中没有查到问题列表')
                raise Exception('mysql中没有数据，请检查')

            # all_questions [467,1]  467代表467行， 1代表只查1列数据

            # TODO 方法1：直接使用循环
            # questions = [question[0] for question in all_questions] 等价于：
            # questions = []
            # for row in all_questions:
            #     # question: [1]
            #     question = row[0]
            #     questions.append(question)
            #
            # TODO 方法2：使用列表表达式
            # 1. for question in all_questions : 拿到每一条数据
            # 2. question[0] 拿到第一列的数据
            questions = [question[0] for question in all_questions]
            tokenizer_questions = [preprocess_text(question[0]) for question in all_questions]

            # 把数据存到redis
            self.redis_client.set_data(question_key, questions)
            self.redis_client.set_data(tokenized_question_key, tokenizer_questions)

            # 赋值到内存，准备构建bm25检索器
            self.tokenizer_questions = tokenizer_questions
            self.questions = questions

        self.logger.info(f'加载问题：{self.questions}')
        self.logger.info(f'加载分词后的问题：{self.tokenizer_questions}')

        # 2.4 初始化bm25模型，并传入：分词问题列表
        self.bm25 = BM25Okapi(self.tokenizer_questions)

    """
    需求：接收query，计算BM25分值，并返回给用户匹配到的答案
    实现：对应_softmax()和search()方法
    思路步骤：
    1. 直接查询缓存，看是否有完全一致的问题的答案缓存
    2. 对query进行分词
    3. 通过BM25模型计算文档匹配分数
    4*. 对分数进行归一化(需要实现softmax公式)
    5. 获得分数最高的文档，并判断是否超过给定阈值
    6*. 查询文档对应的答案，并缓存到redis中
    """

    def _softmax(self, scores):
        """
        把一个一维向量的数据进行归一化， 让所有元素之和 = 1
        :param scores: 文档分数， query和467个文档的相似度 scores: [467] 一维向量
        :return: 归一化以后得向量 [467]
        """
        # scores长度 = 文档的个数
        # TODO 作用 (0，正无穷) -> 归一化 -> [0,1] , 所有元素之和=1。 相当于概率分布
        # 用它是因为我们要判断匹配以后的阈值是否超过0.85
        # [6,12,6 ] -> [0.25,0.5,0.25] (示意， 实际计算结果不是)
        # 计算 Softmax 分数
        # TODO scores - np.max(scores) 的作用是为了缩小分数的数值大小，减少指数函数对结果的影响 [ 0.1, 0.9] -> [0.3 ,0.7]
        # scores: 向量， np.max(scores)：最大值（单值）
        # scores: [score1, score2, score3...]
        # max_score : np.max(scores)
        # TODO 广播机制
        # scores - np.max(scores): [score1 - max_score, score2- max_score, score3 - max_score...]
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        # exp_scores：所有的
        # exp_scores：[exp_score1, exp_score2 ,exp_score3]
        # max_exp_score: exp_scores.sum()
        # exp_scores / exp_scores.sum() :  [exp_score1 / max_exp_score, exp_score2 /max_exp_score  ,exp_score3/max_exp_score]
        return exp_scores / exp_scores.sum()

    # TODO 查询场景梳理：
    # 1. query异常，不进行处理，不调用RAG系统; 2. query正常，阈值不满足, 调用RAG系统 3. query正常，阈值满足，查不到【redis、mysql查不到，执行异常】，调用RAG系统
    # 4. query正常，阈值满足, 查得到【redis、mysql任意】，直接返回，不再调用RAG系统
    def search(self, query, threshold=0.85):
        """
            实现bm25检索
        :param query: 用户的问题
        :param threshold: 可靠答案阈值
        :return: (答案，是否要调用RAG系统)
        """
        # 对异常情况的判断
        if not query or not isinstance(query, str):
            return None, False
        #  1. 直接查询缓存，看是否有完全一致的问题的答案缓存
        answer = self.redis_client.get_answer(question=query)
        if answer:
            self.logger.info(f'在redis中找到了完全一致的问题，答案：{answer}')
            return answer, False

        #  2. 对query进行分词
        tokenized_query = preprocess_text(query)
        #  3. 通过BM25模型计算文档匹配分数
        # scores: [467], tokenized_query 和467个文档的分数列表
        scores = self.bm25.get_scores(tokenized_query)
        #  4*. 对分数进行归一化(需要实现softmax公式)
        softmax_scores = self._softmax(scores)
        #  5. 获得分数最高的文档，并判断是否超过给定阈值
        max_index = np.argmax(softmax_scores)
        max_score = np.max(softmax_scores)
        if max_score >= threshold:
            self.logger.info(f'匹配问题成功，分数:{max_score}')
            question = self.questions[max_index]
            # 尝试在redis中查询对应的问题
            try:
                answer = self.redis_client.get_answer(question=question)
                if answer:
                    self.logger.info(f'在redis找到的了对应的问题，答案：{answer}')
                    return answer, False
                # 1. query异常，不进行处理，不调用RAG系统; 2. query正常，阈值不满足, 调用RAG系统 3. query正常，阈值满足，查不到【redis、mysql查不到，执行异常】，调用RAG系统
                # 4. query正常，阈值满足, 查得到【redis、mysql任意】，直接返回，不再调用RAG系统
                #  6*. 查询文档对应的答案，并缓存到redis中
                # 查询mysql
                answer = self.mysql_client.fetch_answer(question=question)
                if answer:
                    self.logger.info(f'在mysql找到了对应的问题，答案：{answer}')
                    self.redis_client.set_answer(question=question, answer=answer)
                    return answer, False
                return None, True
            except Exception as e:
                self.logger.error(f'查询过程遇到了问题: {e}')
                return None, True
        self.logger.info(f'匹配失败，最高分数:{max_score}')
        return None, True


if __name__ == '__main__':
    mysql_client = MysqlClient()
    redis_client = RedisClient()
    bm25 = BM25Search(mysql_client=mysql_client, redis_client=redis_client)
