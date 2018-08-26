def scrape_bolighed(sleeping):
    import requests, json, tqdm, time
    ### Scrape af 'Bolighed.dk'
    # First we test if we can get some data from Bolighed
    url = 'https://bolighed.dk/api/external/market/propertyforsale/?limit=40&offset=0&view=list&type__in=300&ordering=mtid'
    response = requests.get(url)
    if response.ok:  # response.ok is True if status code is 200
        d = response.json()
    else:
        print('error')

    # Total number of listings from 22.08.2018
    listings = d['count'] # = 7855 opslag
    listings
    print('ok')

    # Collect all links from search on Bolighed by changing the page-parameter
    links = []
    for offset in range(0,listings+40,40):
        url = 'https://bolighed.dk/api/external/market/propertyforsale/?limit=40&offset={o}&view=list&type__in=300&ordering=mtid'.format(o = offset)
        links.append(url)
    len(links)

    # The scraping part - OBS only run this once. It takes almost 20 minutes.
    done = set()
    data = []

    for url in tqdm.tqdm(links):
        response = requests.get(url)

        if response.ok:
            d = response.json()
        else:
            print('error')

        data += d['results']
        time.sleep(sleeping)

    # Save our collected data
    return data
