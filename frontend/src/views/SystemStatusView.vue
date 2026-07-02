<template>
  <section>
    <div class="toolbar">
      <div>
        <strong>系统状态</strong>
        <p class="muted">MySQL、Redis、Milvus、BERT、BGE-M3、Qwen、RAGAS</p>
      </div>
      <el-button type="primary" @click="load">刷新</el-button>
    </div>
    <div class="metric-grid">
      <div class="panel metric">
        <span class="muted">MySQL</span>
        <strong>{{ ok(status.mysql?.ok) }}</strong>
      </div>
      <div class="panel metric">
        <span class="muted">Redis</span>
        <strong>{{ ok(status.redis?.ok) }}</strong>
      </div>
      <div class="panel metric">
        <span class="muted">Milvus</span>
        <strong>{{ ok(status.milvus?.ok) }}</strong>
      </div>
      <div class="panel metric">
        <span class="muted">RAGAS</span>
        <strong>{{ ok(status.models?.ragas?.available) }}</strong>
      </div>
    </div>
    <el-card shadow="never">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="应用">{{ status.app_name }}</el-descriptions-item>
        <el-descriptions-item label="语料版本">{{ status.corpus_version }}</el-descriptions-item>
        <el-descriptions-item label="BERT 路径">{{ status.models?.bert?.configured_path }}</el-descriptions-item>
        <el-descriptions-item label="BERT 可用">{{ ok(status.models?.bert?.available) }}</el-descriptions-item>
        <el-descriptions-item label="Embedding">{{ status.models?.embedding?.name }}</el-descriptions-item>
        <el-descriptions-item label="Reranker">{{ status.models?.reranker?.name }}</el-descriptions-item>
        <el-descriptions-item label="Qwen 模型">{{ status.models?.qwen?.model }}</el-descriptions-item>
        <el-descriptions-item label="Qwen Key">{{ ok(status.models?.qwen?.api_configured) }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from '../api'

const status = ref<Record<string, any>>({})

async function load() {
  const { data } = await api.get('/api/system/status')
  status.value = data
}

function ok(value: unknown) {
  return value ? '正常' : '未就绪'
}

onMounted(load)
</script>
