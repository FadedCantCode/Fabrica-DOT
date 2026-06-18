# DOT evolving-neuron

把 evolving_neuron.py 的自我演化神經元,改成可以掛在 Vercel 上、靠排程器自己跑的版本。

## 先說清楚 Vercel 實際上是怎麼運作的

Vercel 沒有「一直在背景跑的程式」這種東西。`/api/evolve` 是一個 serverless
function,每次被呼叫才會啟動、跑完就結束,呼叫之間完全不會記得任何東西。
「自己持續進化」的感覺,是靠排程器(cron 或外部服務)定期去呼叫這個
endpoint 做出來的;每次呼叫的流程是:

1. 從 Upstash Redis 讀回上次存的族群狀態
2. 跑一批世代(預設 50 代,可用 `?generations=100` 調整,上限 500)
3. 把新狀態存回 Redis
4. 回傳目前進度的 JSON

所以「狀態存在哪裡」比「程式在哪裡跑」更重要——這也是下面唯一一個我沒辦法
幫你自動做完的步驟。

## 部署步驟

我這邊的執行環境沒有網路存取權限,沒辦法直接幫你跑 `vercel deploy`,
下面這幾行你在自己的終端機跑就好:

```bash
npm install -g vercel        # 如果還沒裝過
cd evolving-neuron-vercel
vercel link                  # 會問你要建立新專案還是連到既有專案
vercel deploy --prod
```

部署完會拿到一個 `https://你的專案.vercel.app` 網址,直接瀏覽器打開
`https://你的專案.vercel.app/api/evolve` 應該會看到類似這樣的 JSON
(這時候 `redis_connected` 會是 `false`,因為還沒接 Redis,正常):

```json
{"redis_connected": false, "total_generations": 50, "current_task": "a", ...}
```

## 接上持久化儲存(唯一需要在 Vercel 網頁上手動點的步驟)

1. 進你的 Vercel 專案 → **Storage** 頁籤 → **Marketplace** → 找 **Upstash**
   → 選 **Redis** → 建立一個免費的資料庫並連結到這個專案
2. Vercel 會自動把連線資訊寫成環境變數(`UPSTASH_REDIS_REST_URL` /
   `UPSTASH_REDIS_REST_TOKEN`,舊版叫 `KV_REST_API_URL` /
   `KV_REST_API_TOKEN`,程式碼兩種命名都認)
3. 重新部署一次讓新的環境變數生效:`vercel deploy --prod`
4. 再打一次 `/api/evolve`,這次 `redis_connected` 應該是 `true`,
   而且連續打幾次會看到 `total_generations` 持續累加,代表狀態真的接上了

## 排程頻率怎麼選

`vercel.json` 裡預設用 Vercel 內建 cron,每天跑一次(UTC 03:00),這是
Hobby(免費)方案能設定的上限——Hobby 的 cron 一天只能觸發一次,排得更密
deploy 會直接失敗。三個選項:

- **維持每天一次**:什麼都不用改,但因為一天才跑一批,可以把
  `GENERATIONS_PER_RUN_DEFAULT`(在 `api/evolve.py` 裡)調高一點,
  讓單次累積的進化量大一些。
- **升級 Pro($20/月)**:`vercel.json` 的 `schedule` 可以改成
  `"*/10 * * * *"` 之類更密的頻率,不受每天一次限制。
- **不升級,改用外部排程器**:`/api/evolve` 本身就是一個普通的 HTTPS
  endpoint,任何排程服務(例如免費的 cron-job.org)都可以用任意頻率去
  打它,跟 Vercel 自己的 cron 限制完全無關。這個情況下可以直接把
  `vercel.json` 裡的 `crons` 整段拿掉,只留外部排程器設定。

## 安全性(可選但建議)

如果不想讓網路上隨便一個知道網址的人也能觸發演化(會浪費 function 用量、
也可能跟你自己的排程同時寫入造成 race condition),在 Vercel 專案的
環境變數加一個 `CRON_SECRET`(隨機字串即可)。Vercel 自己的 cron 呼叫會
自動帶上對應的 Authorization header;如果用外部排程器,要手動在它的設定
裡加上 `Authorization: Bearer <你的CRON_SECRET>` 這個 header。

## 我本地測試過的部分,沒測過的部分

用假的 Redis(本機檔案模擬)測過完整的「讀狀態 → 跑一批 → 存狀態」流程,
包括跨多次獨立 process 呼叫狀態有沒有正確累積、任務切換時機對不對、
continual learning 保護有沒有在切換後正確啟動——這些都驗證沒問題。

沒測過、也測不了的部分:真正打 Upstash 的 HTTPS REST API(我的執行環境
沒有網路),以及 Vercel 自己的 cron 實際觸發時機。這兩塊照官方文件寫的,
邏輯上應該沒問題,但要等你真的部署接上 Redis 之後才能 100% 確認。
