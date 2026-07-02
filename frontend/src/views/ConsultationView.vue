<template>
  <div class="consultation">
    <section class="panel chat-panel">
      <div class="chat-header">
        <div>
          <strong>法律咨询</strong>
          <p class="muted">BERT 意图分类、FAQ 优先命中、Milvus 混合检索和 Qwen 生成</p>
        </div>
        <el-select v-model="sourceFilter" placeholder="全部领域" clearable style="width: 180px">
          <el-option v-for="source in sources" :key="source" :label="source" :value="source" />
        </el-select>
      </div>
      <div class="messages">
        <div v-for="message in messages" :key="message.id" :class="['message', message.role]">
          <div>{{ message.content }}</div>
        </div>
      </div>
      <div class="composer">
        <el-input v-model="query" type="textarea" :rows="3" resize="none" placeholder="例如：未签劳动合同有哪些法律风险？" />
        <el-button type="primary" :loading="loading" @click="send">发送</el-button>
      </div>
    </section>
    <aside class="side-stack">
      <section class="panel side-card">
        <strong>链路状态</strong>
        <el-steps direction="vertical" :active="activeStep" finish-status="success">
          <el-step v-for="step in steps" :key="step" :title="step" />
        </el-steps>
      </section>
      <section class="panel side-card">
        <strong>答案元数据</strong>
        <el-descriptions :column="1" size="small" border>
          <el-descriptions-item label="类型">{{ lastEnd.answer_type || '-' }}</el-descriptions-item>
          <el-descriptions-item label="路由">{{ lastEnd.route || '-' }}</el-descriptions-item>
          <el-descriptions-item label="语料版本">{{ lastEnd.corpus_version || '-' }}</el-descriptions-item>
        </el-descriptions>
      </section>
      <section class="panel side-card">
        <strong>引用来源</strong>
        <el-empty v-if="!references.length" description="暂无引用" :image-size="64" />
        <el-scrollbar v-else height="260px">
          <div v-for="(ref, index) in references" :key="index" class="reference">
            <b>{{ ref.title || ref.chunk_id || '引用片段' }}</b>
            <p class="muted">{{ ref.source_url || '未提供 source_url' }}</p>
            <p class="muted">retrieval: {{ score(ref.retrieval_score) }} · rerank: {{ score(ref.rerank_score) }}</p>
          </div>
        </el-scrollbar>
      </section>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, createSession, streamQuestion, type ChatEndPayload, type Reference } from '../api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

const sessionId = ref('')
const query = ref('')
const loading = ref(false)
const sourceFilter = ref('')
const sources = ref<string[]>([])
const references = ref<Reference[]>([])
const lastEnd = ref<ChatEndPayload>({})
const activeStep = ref(0)
const steps = ['意图分类', 'FAQ/热点检索', 'Query 改写', 'Milvus 混合检索', 'ReRank 精排', 'Qwen 生成']
const messages = ref<Message[]>([
  { id: crypto.randomUUID(), role: 'assistant', content: '您好，我是智慧法务大脑。请描述公司法、合同、劳动用工或合规风险相关问题。' }
])

onMounted(async () => {
  sessionId.value = await createSession()
  const { data } = await api.get('/api/sources')
  sources.value = data.sources || []
})

function send() {
  const text = query.value.trim()
  if (!text || loading.value) return
  query.value = ''
  loading.value = true
  activeStep.value = 1
  references.value = []
  lastEnd.value = {}
  messages.value.push({ id: crypto.randomUUID(), role: 'user', content: text })
  const assistant: Message = { id: crypto.randomUUID(), role: 'assistant', content: '' }
  messages.value.push(assistant)
  const timer = window.setInterval(() => {
    if (activeStep.value < steps.length) activeStep.value += 1
  }, 700)
  streamQuestion(
    { query: text, source_filter: sourceFilter.value || null, session_id: sessionId.value },
    {
      onToken: (token) => {
        assistant.content += token
      },
      onEnd: (payload) => {
        window.clearInterval(timer)
        activeStep.value = steps.length
        loading.value = false
        lastEnd.value = payload
        references.value = payload.references || []
      },
      onError: (message) => {
        window.clearInterval(timer)
        loading.value = false
        assistant.content = message
      }
    }
  )
}

function score(value?: number) {
  return typeof value === 'number' ? value.toFixed(3) : '-'
}
</script>

<style scoped>
.consultation {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 16px;
  height: calc(100vh - 88px);
}

.chat-panel {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 0;
}

.chat-header,
.composer {
  padding: 14px;
  border-bottom: 1px solid #e4e7ec;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.composer {
  border-bottom: 0;
  border-top: 1px solid #e4e7ec;
}

.messages {
  padding: 16px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.message {
  max-width: 82%;
  padding: 12px 14px;
  border-radius: 8px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.message.user {
  align-self: flex-end;
  background: #e6f4ff;
}

.message.assistant {
  align-self: flex-start;
  background: #f2f4f7;
}

.side-stack {
  display: grid;
  gap: 12px;
  align-content: start;
}

.side-card {
  padding: 14px;
}

.reference {
  border-bottom: 1px solid #edf0f5;
  padding: 10px 0;
}

@media (max-width: 1100px) {
  .consultation {
    grid-template-columns: 1fr;
    height: auto;
  }
}
</style>
