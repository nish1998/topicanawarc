# Topicana
Fetching data through commoncrawl and topic modelling using nytimes and other news data. 

## To run in local

### Change app.py code:
```
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run()
```

### To:
```
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    #port = int(os.environ.get('PORT', 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run()
```    
### Run: $ python app.py

Hosted on heroku: https://unfoundtopicana.herokuapp.com/
