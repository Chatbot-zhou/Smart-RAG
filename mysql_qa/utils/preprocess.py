# 导入分词库
import jieba
# 导入日志
from base.logger import single_logger


# 调用： 1.构建bm25的文档数据集的时候 2.用户的query传入的时候
def preprocess_text(text):
    # 预处理文本
    single_logger.debug(f"开始预处理文本:{text}")
    try:
        # 分词并转换为小写
        # TODO 这里为什么要转小写?
        # 1. bm25算法基于单词的词频统计做相似度判断，使用的 单词1==单词2 的判断逻辑 【java Java JAVA】
        # 2. 对于 【java Java JAVA】做一个统一的处理 -> java
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        single_logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    print(preprocess_text("PYTHON是世界上最好的语言"))
