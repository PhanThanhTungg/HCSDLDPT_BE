export const getIndex = (req, res) => {
  res.render('index', { title: 'Express' , message: 'Hello World!' });
}