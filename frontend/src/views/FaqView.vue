<template>
  <section>
    <div class="toolbar">
      <div>
        <strong>FAQ 管理</strong>
        <p class="muted">MySQL 存储 FAQ，Redis 只保存热点索引</p>
      </div>
      <el-button type="primary" @click="load">刷新</el-button>
    </div>
    <el-card shadow="never" class="import-card">
      <el-form :inline="true" @submit.prevent>
        <el-form-item label="数据集路径">
          <el-input v-model="datasetPath" placeholder="data/faq/legal_faq.jsonl" style="width: 360px" />
        </el-form-item>
        <el-form-item>
          <el-button :loading="importing" @click="importFaq">导入 FAQ</el-button>
        </el-form-item>
      </el-form>
    </el-card>
    <el-table :data="items" border height="calc(100vh - 250px)" class="panel">
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="original_question" label="问题" min-width="260" show-overflow-tooltip />
      <el-table-column prop="category" label="分类" width="120" />
      <el-table-column prop="source_type" label="来源类型" width="110" />
      <el-table-column prop="review_status" label="审核" width="100" />
      <el-table-column prop="hit_count" label="命中" width="90" />
      <el-table-column prop="is_hot" label="热点" width="80">
        <template #default="{ row }">
          <el-tag :type="row.is_hot ? 'success' : 'info'" size="small">{{ row.is_hot ? '是' : '否' }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="faq_version" label="FAQ 版本" width="140" />
      <el-table-column prop="corpus_version" label="语料版本" width="160" />
    </el-table>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'

const items = ref([])
const datasetPath = ref('')
const importing = ref(false)

async function load() {
  const { data } = await api.get('/api/faqs')
  items.value = data.items || []
}

async function importFaq() {
  if (!datasetPath.value.trim()) return
  importing.value = true
  try {
    const { data } = await api.post('/api/faqs/import', { dataset_path: datasetPath.value.trim() })
    ElMessage.success(`已导入 ${data.imported} 条 FAQ`)
    await load()
  } finally {
    importing.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.import-card {
  margin-bottom: 12px;
}
</style>
