<template>
  <section>
    <div class="toolbar">
      <div>
        <strong>RAGAS 评估</strong>
        <p class="muted">展示真实评估记录，指标由后端评估脚本写入 MySQL</p>
      </div>
      <el-button type="primary" @click="load">刷新</el-button>
    </div>
    <el-table :data="items" border class="panel">
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="run_name" label="运行名称" min-width="180" />
      <el-table-column prop="dataset_path" label="数据集" min-width="220" show-overflow-tooltip />
      <el-table-column prop="corpus_version" label="语料版本" width="160" />
      <el-table-column prop="metrics_json" label="指标" min-width="260" show-overflow-tooltip />
      <el-table-column prop="created_at" label="时间" width="180" />
    </el-table>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from '../api'

const items = ref([])

async function load() {
  const { data } = await api.get('/api/evaluations/ragas')
  items.value = data.items || []
}

onMounted(load)
</script>
