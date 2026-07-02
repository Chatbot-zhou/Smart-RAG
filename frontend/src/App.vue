<template>
  <el-container class="shell">
    <el-aside width="248px" class="sidebar">
      <div class="brand">
        <strong>智慧法务大脑</strong>
        <span>企业法务 RAG 工作台</span>
      </div>
      <el-menu router :default-active="$route.path" class="nav">
        <el-menu-item index="/consultation">法律咨询台</el-menu-item>
        <el-menu-item index="/faqs">FAQ 管理</el-menu-item>
        <el-menu-item index="/corpus">知识库管理</el-menu-item>
        <el-menu-item index="/evaluations">RAGAS 评估</el-menu-item>
        <el-menu-item index="/status">系统状态</el-menu-item>
        <el-menu-item index="/agents">Agent 占位</el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="topbar">
        <span>{{ pageTitle }}</span>
        <el-tag type="success" effect="plain">RAG 主链路</el-tag>
      </el-header>
      <el-main class="content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const titles: Record<string, string> = {
  '/consultation': '法律咨询台',
  '/faqs': 'FAQ 管理',
  '/corpus': '知识库管理',
  '/evaluations': 'RAGAS 评估',
  '/status': '系统状态',
  '/agents': 'Agent 占位'
}
const pageTitle = computed(() => titles[route.path] || '智慧法务大脑')
</script>
