import * as controller from './controller.js';

export default (app)=>{
  app.get('/', controller.getIndex);
}