import { createRouter, createWebHistory } from 'vue-router'
import ConsultationView from './views/ConsultationView.vue'
import FaqView from './views/FaqView.vue'
import CorpusView from './views/CorpusView.vue'
import EvaluationView from './views/EvaluationView.vue'
import SystemStatusView from './views/SystemStatusView.vue'
import AgentPlaceholderView from './views/AgentPlaceholderView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/consultation' },
    { path: '/consultation', component: ConsultationView },
    { path: '/faqs', component: FaqView },
    { path: '/corpus', component: CorpusView },
    { path: '/evaluations', component: EvaluationView },
    { path: '/status', component: SystemStatusView },
    { path: '/agents', component: AgentPlaceholderView }
  ]
})
