const { Telegraf } = require("telegraf");
const puppeteer = require("puppeteer");
const fs = require("fs").promises;
const path = require("path");
const { OpenAI } = require("openai");
const { loadLatokenManualData } = require("./addManualData");

// Конфигурация
const TELEGRAM_TOKEN = "";
const OPENAI_API_KEY = "";

// Инициализация клиентов
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

const bot = new Telegraf(TELEGRAM_TOKEN);

// Пути для сохранения данных
const DATA_DIR = path.join(__dirname, "data");
const CACHE_DIR = path.join(__dirname, "cache");

// Функция задержки для контроля нагрузки
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Класс для кэширования данных
class DataCache {
  constructor(cacheDir = CACHE_DIR) {
    this.cacheDir = cacheDir;
  }

  async initialize() {
    try {
      await fs.mkdir(this.cacheDir, { recursive: true });
    } catch (error) {
      if (error.code !== "EEXIST") {
        console.error("Failed to create cache directory:", error);
        throw error;
      }
    }
  }

  generateCacheKey(url) {
    return url
      .replace(/^https?:\/\//, "")
      .replace(/[^a-zA-Z0-9]/g, "_")
      .substring(0, 100);
  }

  async getFromCache(url) {
    try {
      const cacheKey = this.generateCacheKey(url);
      const filePath = path.join(this.cacheDir, `${cacheKey}.json`);

      const data = await fs.readFile(filePath, "utf8");
      const cachedData = JSON.parse(data);

      // Проверяем срок действия кэша (24 часа)
      const cacheAge = Date.now() - cachedData.timestamp;
      if (cacheAge > 24 * 60 * 60 * 1000) {
        console.log(`Cache for ${url} is expired`);
        return null;
      }

      console.log(`Cache hit for ${url}`);
      return cachedData.data;
    } catch (error) {
      // Если файл не найден или не может быть прочитан, считаем это промахом кэша
      return null;
    }
  }

  async saveToCache(url, data) {
    try {
      const cacheKey = this.generateCacheKey(url);
      const filePath = path.join(this.cacheDir, `${cacheKey}.json`);

      const cachedData = {
        url,
        timestamp: Date.now(),
        data,
      };

      await fs.writeFile(filePath, JSON.stringify(cachedData), "utf8");
      console.log(`Cached data for ${url}`);
    } catch (error) {
      console.error(`Failed to cache data for ${url}:`, error);
    }
  }
}

// Класс для векторного поиска на основе косинусного сходства
class VectorSearch {
  constructor() {
    this.vectors = [];
    this.ids = [];
  }

  // Метод для добавления векторов
  add(vectors, ids = null) {
    if (!vectors || vectors.length === 0) return;

    for (let i = 0; i < vectors.length; i++) {
      this.vectors.push(vectors[i]);
      this.ids.push(ids ? ids[i] : this.vectors.length - 1);
    }

    console.log(
      `Added ${vectors.length} vectors, total: ${this.vectors.length}`
    );
  }

  // Рассчет косинусного сходства между двумя векторами
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // Поиск ближайших соседей
  search(queryVector, k = 3) {
    if (this.vectors.length === 0) {
      return [];
    }

    // Рассчитываем сходство со всеми векторами
    const similarities = this.vectors.map((vector, index) => ({
      id: this.ids[index],
      similarity: this.cosineSimilarity(queryVector, vector),
    }));

    // Сортируем по убыванию сходства и берем топ-k
    const topK = similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);

    return topK.map((item) => item.id);
  }
}

// Класс для управления базой знаний
class KnowledgeBase {
  constructor() {
    this.data = [];
    this.embeddings = []; // Кэш эмбеддингов
    this.vectorSearch = new VectorSearch(); // Индекс для векторного поиска
    this.initialized = false;
    this.dimension = 1536; // Размерность для OpenAI text-embedding-3-small
  }

  async initialize() {
    if (this.initialized) return;

    try {
      console.log("Initializing knowledge base...");

      // Создание директории для данных
      try {
        await fs.mkdir(DATA_DIR, { recursive: true });
      } catch (err) {
        if (err.code !== "EEXIST") throw err;
      }

      // Проверка существующих данных
      const loaded = await this.load();

      if (!loaded) {
        console.log("No existing knowledge base found, creating a new one");
        this.data = [];
        this.embeddings = [];
        this.vectorSearch = new VectorSearch();
      } else {
        console.log(
          `Loaded existing knowledge base with ${this.data.length} documents`
        );
      }

      this.initialized = true;
      console.log("Knowledge base initialization complete");
    } catch (error) {
      console.error("Error initializing knowledge base:", error);
      // Fallback to empty base
      this.data = [];
      this.embeddings = [];
      this.vectorSearch = new VectorSearch();
      this.initialized = true;
    }
  }

  async getEmbedding(text) {
    try {
      // Используем OpenAI API для получения эмбеддингов
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
        encoding_format: "float",
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error("Error generating embedding with OpenAI:", error);
      throw error;
    }
  }

  async addDocument(text, source) {
    await this.initialize();

    if (!text || !text.trim()) {
      console.log("Empty text, skipping");
      return;
    }

    // Разделение текста на чанки примерно по 8000 символов
    const chunks = [];
    const maxLength = 8000;

    for (let i = 0; i < text.length; i += maxLength) {
      const chunk = text.substring(i, i + maxLength);
      if (chunk.trim()) {
        chunks.push(chunk);
      }
    }

    const startIndex = this.data.length;

    // Сохраняем документы
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      if (chunk.trim()) {
        this.data.push({
          text: chunk,
          source: source,
        });
      }
    }

    // Если есть новые документы, получаем их эмбеддинги
    if (this.data.length > startIndex) {
      try {
        // Получаем эмбеддинги для новых документов
        const newTexts = this.data.slice(startIndex).map((item) => item.text);

        for (const text of newTexts) {
          try {
            const embedding = await this.getEmbedding(text);
            this.embeddings.push(embedding);

            // Добавляем в индекс
            this.vectorSearch.add([embedding], [this.embeddings.length - 1]);
          } catch (embeddingError) {
            console.error(
              "Error getting embedding for document:",
              embeddingError
            );
          }

          // Делаем небольшую паузу между запросами к API
          await new Promise((resolve) => setTimeout(resolve, 200));
        }
      } catch (error) {
        console.error("Error processing new documents:", error);
      }
    }
  }

  async buildIndex() {
    await this.initialize();

    if (this.data.length === 0) {
      console.log("No data to build index");
      return;
    }

    try {
      console.log(`Building index for ${this.data.length} documents...`);

      // Очищаем имеющиеся индексы
      this.embeddings = [];
      this.vectorSearch = new VectorSearch();

      // Обрабатываем документы небольшими группами
      const batchSize = 5;

      for (let i = 0; i < this.data.length; i += batchSize) {
        const batch = this.data.slice(i, i + batchSize);
        console.log(
          `Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(
            this.data.length / batchSize
          )}`
        );

        try {
          // Получаем эмбеддинги для документов в пакете
          for (const item of batch) {
            try {
              const embedding = await this.getEmbedding(item.text);
              this.embeddings.push(embedding);

              // Добавляем в индекс
              this.vectorSearch.add([embedding], [this.embeddings.length - 1]);
            } catch (embeddingError) {
              console.error(
                "Error getting embedding for document:",
                embeddingError
              );
            }

            // Делаем небольшую паузу между запросами к API
            await new Promise((resolve) => setTimeout(resolve, 200));
          }
        } catch (batchError) {
          console.error(
            `Error processing batch ${Math.floor(i / batchSize) + 1}:`,
            batchError
          );
        }

        // Делаем паузу между батчами
        if (i + batchSize < this.data.length) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }

      console.log(
        `Index built successfully with ${this.data.length} documents`
      );
      // Save the index immediately after building it
      await this.save();
    } catch (error) {
      console.error("Error building index:", error);
      throw error;
    }
  }

  async search(query, k = 3) {
    await this.initialize();

    if (this.data.length === 0) {
      console.log("Knowledge base is empty");
      return [];
    }

    // Проверка наличия эмбеддингов
    if (this.embeddings.length === 0) {
      console.log("No embeddings available for search");
      return [];
    }

    try {
      // Получение эмбеддинга для запроса
      const queryEmbedding = await this.getEmbedding(query);

      // Поиск ближайших соседей
      const resultIndices = this.vectorSearch.search(queryEmbedding, k);

      // Проверяем наличие результатов
      if (!resultIndices || resultIndices.length === 0) {
        console.log("No search results found");
        return [];
      }

      console.log(`Found ${resultIndices.length} results:`, resultIndices);

      // Формируем результаты
      const searchResults = [];
      for (const idx of resultIndices) {
        // Проверяем, что индекс валидный
        if (idx >= 0 && idx < this.data.length) {
          searchResults.push(this.data[idx]);
        }
      }

      return searchResults;
    } catch (error) {
      console.error("Error during search:", error);
      return [];
    }
  }

  async save() {
    await this.initialize();

    try {
      // Сохраняем данные
      await fs.writeFile(
        path.join(DATA_DIR, "knowledge_base.json"),
        JSON.stringify(this.data),
        "utf8"
      );

      // Сохраняем эмбеддинги
      await fs.writeFile(
        path.join(DATA_DIR, "embeddings.json"),
        JSON.stringify(this.embeddings),
        "utf8"
      );

      console.log("Knowledge base saved successfully");
      return true;
    } catch (error) {
      console.error("Error saving knowledge base:", error);
      return false;
    }
  }

  async load() {
    try {
      // Проверка существования файлов
      const dataExists = await fs
        .access(path.join(DATA_DIR, "knowledge_base.json"))
        .then(() => true)
        .catch(() => false);

      const embeddingsExist = await fs
        .access(path.join(DATA_DIR, "embeddings.json"))
        .then(() => true)
        .catch(() => false);

      if (!dataExists || !embeddingsExist) {
        console.log("Required knowledge base files not found");
        return false;
      }

      // Загрузка данных
      const dataContent = await fs.readFile(
        path.join(DATA_DIR, "knowledge_base.json"),
        "utf8"
      );
      this.data = JSON.parse(dataContent);

      // Загрузка эмбеддингов
      const embeddingsContent = await fs.readFile(
        path.join(DATA_DIR, "embeddings.json"),
        "utf8"
      );
      this.embeddings = JSON.parse(embeddingsContent);

      // Создаем индекс из загруженных эмбеддингов
      this.vectorSearch = new VectorSearch();
      for (let i = 0; i < this.embeddings.length; i++) {
        this.vectorSearch.add([this.embeddings[i]], [i]);
      }

      console.log(
        `Loaded knowledge base with ${this.data.length} documents and ${this.embeddings.length} embeddings`
      );
      return true;
    } catch (error) {
      console.error("Error loading knowledge base:", error);
      return false;
    }
  }
}

// Запуск браузера Puppeteer для работы со страницами
async function launchBrowser() {
  return await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
    ],
  });
}

// Функция для извлечения ссылок со страницы с поддержкой iframe
async function extractLinksFromPage(page, url) {
  console.log(`Извлечение ссылок с ${url}`);

  try {
    // Get all frames (main page and iframes)
    await page.goto(url, { waitUntil: "networkidle2" });
    const frames = page.frames();

    let allLinks = [];

    // Process each frame
    for (const frame of frames) {
      try {
        // Extract links from this frame
        const frameLinks = await frame.evaluate(() => {
          const links = new Set();
          document.querySelectorAll("a").forEach((element) => {
            if (element.href) links.add(element.href);
          });
          return Array.from(links);
        });

        allLinks = [...allLinks, ...frameLinks];
      } catch (error) {
        console.log(`Не удалось получить ссылки из фрейма: ${error.message}`);
      }
    }

    // Filter for relevant links
    const relevantLinks = allLinks.filter((link) => {
      return (
        link.includes("coda.io") &&
        (link.includes("latoken") || link.includes("talent"))
      );
    });

    console.log(
      `Найдено ${relevantLinks.length} релевантных ссылок на странице ${url}`
    );
    return relevantLinks;
  } catch (error) {
    console.error(`Ошибка при извлечении ссылок с ${url}:`, error.message);
    return [];
  }
}

// Функция для сбора контента со страницы
async function extractContentFromPage(page, url, dataCache) {
  // Проверяем кэш
  const cachedData = await dataCache.getFromCache(url);
  if (cachedData) {
    return cachedData;
  }

  console.log(`Извлечение контента с ${url}`);

  try {
    // Навигация к странице
    await page.goto(url, { waitUntil: "networkidle2", timeout: 30000 });

    // Ждем, чтобы контент загрузился
    await delay(2000);

    // Получаем заголовок и содержимое страницы
    const pageData = await page.evaluate(() => {
      // Получаем заголовок
      const title =
        document.title || document.querySelector("h1")?.innerText || "";

      // Удаляем скрипты и стили для получения чистого текста
      const scripts = Array.from(document.querySelectorAll("script, style"));
      scripts.forEach((s) => s.remove());

      // Получаем текст всей страницы
      const content = document.body.innerText.replace(/\s+/g, " ").trim();

      return { title, content };
    });

    // Формируем результат
    const result = {
      url,
      title: pageData.title,
      content: pageData.content,
    };

    // Сохраняем в кэш если есть контент
    if (result.content && result.content.length > 100) {
      await dataCache.saveToCache(url, result);
    }

    return result;
  } catch (error) {
    console.error(`Ошибка при извлечении контента с ${url}:`, error.message);
    return { url, title: "", content: "" };
  }
}

// Объединенная функция для обхода и сбора данных со всех страниц
async function collectAllContent(dataCache, maxDepth = 3) {
  const browser = await launchBrowser();

  try {
    const page = await browser.newPage();

    // Настройка браузера для имитации пользователя
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    );
    await page.setViewport({ width: 1366, height: 768 });

    // URLs для сбора данных
    const startUrls = [
      // "https://deliver.latoken.com/hackathon",
      "https://coda.io/@latoken/latoken-talent",
    ];

    const visitedUrls = new Set();
    const pageQueue = startUrls.map((url) => ({ url, depth: 0 }));
    const results = [];

    console.log("Начинаем сбор данных со страниц Латокен");

    while (pageQueue.length > 0) {
      const { url, depth } = pageQueue.shift();

      if (visitedUrls.has(url) || depth > maxDepth) {
        continue;
      }

      visitedUrls.add(url);

      try {
        // Извлекаем контент с текущей страницы
        const pageContent = await extractContentFromPage(page, url, dataCache);

        if (pageContent.content && pageContent.content.length > 100) {
          results.push(pageContent);
          console.log(
            `[${results.length}] Добавлена страница: ${pageContent.title}`
          );
        }

        // Извлекаем ссылки только если не достигли максимальной глубины
        if (depth < maxDepth) {
          const links = await extractLinksFromPage(page, url);

          // Добавляем новые ссылки в очередь
          for (const link of links) {
            if (!visitedUrls.has(link)) {
              pageQueue.push({ url: link, depth: depth + 1 });
            }
          }
        }
      } catch (error) {
        console.error(`Ошибка при обработке ${url}:`, error.message);
        // Продолжаем с другими URL
      }

      // Добавляем задержку между запросами
      await delay(2000);
    }

    console.log(`Извлечено ${results.length} страниц с контентом`);
    return results;
  } catch (error) {
    console.error("Ошибка при сборе данных:", error);
    return [];
  } finally {
    await browser.close();
  }
}

// Функция для извлечения всех данных и создания базы знаний
async function extractAllData() {
  // Инициализируем кэш
  const dataCache = new DataCache();
  await dataCache.initialize();

  const kb = new KnowledgeBase();
  await kb.initialize();

  // Проверка существования индекса
  const loaded = await kb.load();
  if (loaded) {
    console.log("Loaded existing knowledge base");
    return kb;
  }

  console.log("Building new knowledge base...");

  await loadLatokenManualData(dataCache, kb);
  console.log("Manually added LATOKEN content from Canva presentations");

  // Сбор данных со всех страниц
  try {
    console.log("Starting content collection...");

    // Получаем содержимое всех страниц
    const allPages = await collectAllContent(dataCache);

    // Добавляем страницы в базу знаний
    for (const page of allPages) {
      if (!page.content || page.content.length < 100) continue;

      // Все документы имеют обобщенную категорию
      const category = "general_info";

      // Добавляем документ в базу знаний
      await kb.addDocument(page.content, category);
      console.log(`Added page "${page.title}" as ${category}`);
    }
  } catch (error) {
    console.error("Error during content collection:", error);
    // Продолжаем даже при ошибке
  }

  // Построение индекса и сохранение
  console.log("Building index...");
  await kb.buildIndex();
  await kb.save();
  console.log("Knowledge base built and saved successfully");

  return kb;
}

// RAG функции
async function queryGPT(query, context = "") {
  const prompt = `
    Ты помощник для кандидатов, желающих присоединиться к компании Латокен. 
    Используй следующий контекст для ответа на вопрос кандидата:
    
    ${context}
    
    Вопрос кандидата: ${query}
    
    Ответь конкретно, предоставь полезную информацию о Латокен и процессе собеседования.
  `;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "Ты полезный ассистент Латокен, который помогает кандидатам.",
        },
        { role: "user", content: prompt },
      ],
      max_tokens: 1000,
      temperature: 0.7,
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error querying GPT:", error);
    return "Извините, произошла ошибка при обработке вашего запроса.";
  }
}

async function generateTestQuestion(context) {
  const prompt = `
    На основе следующего контекста о Латокен, сгенерируй один вопрос с множественным выбором 
    для тестирования знаний кандидата о компании, культуре или процессе собеседования.
    
    Контекст:
    ${context}
    
    Формат ответа:
    Вопрос: [вопрос]
    A. [вариант]
    B. [вариант]
    C. [вариант]
    D. [вариант]
    Правильный ответ: [буква]
  `;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "Ты помощник для создания вопросов для тестирования кандидатов Латокен.",
        },
        { role: "user", content: prompt },
      ],
      max_tokens: 500,
      temperature: 0.8,
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error generating test question:", error);
    return "Извините, не удалось сгенерировать тестовый вопрос.";
  }
}

// Обработчик запросов с улучшенным контекстом
async function enhancedQueryHandler(query, kb) {
  try {
    if (!kb || !kb.initialized) {
      console.error("Knowledge base not initialized properly");
      return {
        answer: "Извините, база знаний не инициализирована. Попробуйте позже.",
        sources: [],
      };
    }

    // Поиск релевантных документов
    const relevantDocs = await kb.search(query, 5);

    if (!relevantDocs || relevantDocs.length === 0) {
      console.log("No relevant documents found for query:", query);
      return {
        answer:
          "Я не нашел специфической информации по вашему запросу. Могу я помочь вам с чем-то другим о компании Латокен?",
        sources: [],
      };
    }

    // Группировка документов по источникам
    const docsBySource = {};
    for (const doc of relevantDocs) {
      if (!doc.source) {
        console.warn("Document without source found:", doc);
        continue;
      }

      if (!docsBySource[doc.source]) {
        docsBySource[doc.source] = [];
      }
      docsBySource[doc.source].push(doc.text);
    }

    // Проверяем, есть ли документы после группировки
    if (Object.keys(docsBySource).length === 0) {
      return {
        answer:
          "Я не смог найти подходящую информацию в базе знаний. Попробуйте переформулировать ваш вопрос.",
        sources: [],
      };
    }

    // Формирование структурированного контекста
    let contextText = "";

    for (const source in docsBySource) {
      contextText += `\n### Информация из категории "${source}":\n`;
      contextText += docsBySource[source].join("\n\n");
      contextText += "\n";
    }

    // Добавляем метаданные для GPT
    const enhancedContext = `
Ниже приведена информация о компании Латокен из различных источников, сгруппированная по категориям.
Используй эту информацию для ответа на вопрос кандидата.
${contextText}
    `;

    // Генерация ответа с улучшенным контекстом
    const answer = await queryGPT(query, enhancedContext);

    return {
      answer,
      sources: Object.keys(docsBySource),
    };
  } catch (error) {
    console.error("Error in enhanced query handler:", error);
    return {
      answer:
        "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже или задайте другой вопрос.",
      sources: [],
    };
  }
}

// Глобальная переменная для базы знаний
let knowledgeBase = null;
let testMode = {};

// Инициализация базы знаний
async function initializeKnowledgeBase() {
  if (!knowledgeBase) {
    console.log("Initializing knowledge base...");
    knowledgeBase = await extractAllData();
  }
  return knowledgeBase;
}

// Обработчики команд Telegram
bot.start(async (ctx) => {
  await ctx.reply(
    `Привет! Я бот-помощник для кандидатов Латокен. Спрашивайте меня о компании, процессе собеседования или культуре Латокен.\nСписок команд: \n/test-для включения/отключения режима тестирования`
  );
});

bot.command("test", (ctx) => {
  const userId = ctx.from.id;
  testMode[userId] = !testMode[userId];

  if (testMode[userId]) {
    ctx.reply(
      "Режим тестирования включен. После каждого ответа, я буду задавать вам тестовый вопрос."
    );
  } else {
    ctx.reply("Режим тестирования выключен.");
  }
});

// Обработчик текстовых сообщений
bot.on("text", async (ctx) => {
  const userId = ctx.from.id;
  const query = ctx.message.text;

  // Игнорирование команд
  if (query.startsWith("/")) return;

  try {
    // Отправляем индикатор набора текста
    await ctx.sendChatAction("typing");

    // Инициализация базы знаний
    const kb = await initializeKnowledgeBase();

    // Используем улучшенный обработчик запросов
    const { answer, sources } = await enhancedQueryHandler(query, kb);

    // Отправляем ответ пользователю
    await ctx.reply(answer);

    // Если включен режим тестирования, генерируем тестовый вопрос
    if (testMode[userId]) {
      try {
        // Поиск релевантных документов для генерации вопроса
        const relevantDocs = await kb.search(query, 3);
        if (relevantDocs && relevantDocs.length > 0) {
          const contextText = relevantDocs.map((doc) => doc.text).join("\n\n");

          // Генерация тестового вопроса
          const testQuestion = await generateTestQuestion(contextText);

          // Небольшая задержка перед отправкой тестового вопроса
          await delay(1000);
          await ctx.reply(
            "\nА теперь давайте проверим ваши знания:\n\n" + testQuestion
          );
        } else {
          console.log(
            "No relevant documents found for test question generation"
          );
        }
      } catch (testError) {
        console.error("Error generating test question:", testError);
      }
    }
  } catch (error) {
    console.error("Error handling message:", error);
    await ctx.reply(
      "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
    );
  }
});

// Запуск бота
async function startBot() {
  try {
    // Инициализация базы знаний перед запуском бота
    await initializeKnowledgeBase();

    // Запуск бота
    await bot.launch();
    console.log("Bot started successfully");
  } catch (error) {
    console.error("Error starting bot:", error);
  }
}

startBot();

// Обработка завершения работы
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
