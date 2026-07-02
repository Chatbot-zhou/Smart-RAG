<template>
  <section>
    <div class="toolbar">
      <div>
        <strong>Agent 下一阶段</strong>
        <p class="muted">本阶段只展示入口，不实现完整合同审查和合同制作 Agent</p>
      </div>
    </div>
    <el-row :gutter="12">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>合同审查 Agent</template>
          <p>下一阶段接入条款抽取、风险评级、法律依据检索、修改建议和审查报告生成。</p>
          <el-tag type="info">planned</el-tag>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>合同制作 Agent</template>
          <p>下一阶段接入模板库、字段填槽、缺失字段追问、条款生成和草案导出。</p>
          <el-tag type="info">planned</el-tag>
        </el-card>
      </el-col>
    </el-row>
  </section>
</template>
