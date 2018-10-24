from bs4 import BeautifulSoup
import urllib
import re

def extract_label(query):
    data = urllib.urlopen('https://mediabiasfactcheck.com/?s='+query).read()
    soup = BeautifulSoup(data,"html.parser")
    # temp = soup.find_all('a')
    matter =  soup.find('article')
    title = matter.find('h3').string
    lable_matter=matter.find('div',{'class':'mh-excerpt'})
    # print re.search('QUESTIONABLE SOURCE',str(lable_matter))
    sources=re.compile('(Satire)|(Questionable Source)|(Conspiracy-Pseudoscience)|(Pro-Science)|(Right Bias)|(Right-Center Bias)|(Least Biased)|(Left-Center Bias)|(Left Bias)',re.IGNORECASE)

    find_label = re.search(sources,str(lable_matter))
    if(find_label):
        return title,find_label.group()
    else:
        return title,''

def main():
    link,label = extract_label("http://www.addictinginfo.org")
    print link,label

if __name__ == '__main__':
    main()
