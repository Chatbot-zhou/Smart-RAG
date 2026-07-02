import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 20000
})

export interface Reference {
  title?: string
  article_no?: string
  source_url?: string
  retrieval_score?: number
  rerank_score?: number
  chunk_id?: string
  parent_id?: string
}

export interface ChatEndPayload {
  references?: Reference[]
  answer_type?: string
  corpus_version?: string
  route?: string
  rewritten_query?: string
  disclaimer?: string
}

export async function createSession() {
  const { data } = await api.post('/api/create_session')
  return data.session_id as string
}

export function streamQuestion(payload: { query: string; source_filter?: string | null; session_id: string }, handlers: {
  onToken: (token: string) => void
  onEnd: (payload: ChatEndPayload) => void
  onError: (message: string) => void
}) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const base = import.meta.env.VITE_WS_BASE || `${protocol}://${window.location.host}`
  const socket = new WebSocket(`${base}/api/stream`)
  socket.onopen = () => socket.send(JSON.stringify(payload))
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type === 'token') handlers.onToken(data.token)
    if (data.type === 'end') {
      handlers.onEnd(data)
      socket.close()
    }
    if (data.type === 'error') {
      handlers.onError(data.error)
      socket.close()
    }
  }
  socket.onerror = () => handlers.onError('WebSocket 连接失败')
  return socket
}
