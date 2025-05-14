import express from 'express';
const app = express();
const port = 3000;

app.set('views', './view')
app.set('view engine', 'pug')

import { fileURLToPath } from 'url';
import { dirname } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
app.use(express.static(`${__dirname}/public`))

import indexRouter from './route.js';
indexRouter(app);

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});