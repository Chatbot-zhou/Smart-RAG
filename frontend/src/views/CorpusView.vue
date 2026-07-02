<template>
  <section>
    <div class="toolbar">
      <div>
        <strong>知识库管理</strong>
        <p class="muted">读取本地官方法规 manifest，不展示 mock 数据</p>
      </div>
      <el-button type="primary" @click="load">刷新</el-button>
    </div>
    <el-alert v-if="!manifest.available" :title="manifest.error || 'manifest 不可用'" type="warning" show-icon class="mb" />
    <div class="metric-grid">
      <div class="panel metric">
        <span class="muted">Manifest 条目</span>
        <strong>{{ manifest.count || 0 }}</strong>
      </div>
      <div class="panel metric">
        <span class="muted">Raw 目录</span>
        <strong class="small">{{ manifest.raw_dir || '-' }}</strong>
      </div>
      <div class="panel metric">
        <span class="muted">Text 目录</span>
        <strong class="small">{{ manifest.text_dir || '-' }}</strong>
      </div>
    </div>
    <el-table :data="manifest.items || []" border height="calc(100vh - 250px)" class="panel">
      <el-table-column prop="title" label="法规标题" min-width="220" show-overflow-tooltip />
      <el-table-column prop="domain" label="领域" width="120" />
      <el-table-column prop="authority" label="发布机关" width="180" />
      <el-table-column prop="status" label="效力" width="100" />
      <el-table-column prop="effective_date" label="施行日期" width="120" />
      <el-table-column prop="sha256" label="sha256" min-width="240" show-overflow-tooltip />
    </el-table>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from '../api'

const manifest = ref<Record<string, any>>({})

async function load() {
  const { data } = await api.get('/api/corpus/manifest')
  manifest.value = data
}

onMounted(load)
</script>

<style scoped>
.mb {
  margin-bottom: 12px;
}

.small {
  font-size: 12px;
  word-break: break-all;
}
</style>
