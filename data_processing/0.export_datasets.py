from bson.json_util import dumps
from pymongo import MongoClient
import json

# Apache
# Hyperledger
# IntelDAOS
# JFrog
# Jira
# JiraEcosystem
# MariaDB
# Mindville
# Mojang
# MongoDB
# Qt
# RedHat
# Sakai
# SecondLife
# Sonatype
# Spring

if __name__ == '__main__':

    client = MongoClient()
    db = client.JiraRepos

    tables = [
        "JFrog",
        "Apache",
        "Hyperledger",
        "IntelDAOS",
        "Jira",
        "JiraEcosystem",
        "MariaDB",
        "Mindville",
        "Mojang",
        "MongoDB",
        "Qt",
        "RedHat",
        "Sakai",
        "SecondLife",
        "Sonatype",
        "Spring",
              ]

    for _table in tables:
        print("_table : ", _table)
        collection = db[_table]
        # file_issue_links = open('./data/public_jira_json/' + _table + '_issue_links_time.json', 'w')
        with open('./data/public_jira_json/' + _table + '_without_issue_link.json', 'w') as file:
            n = 0
            m = 0
            for cursor in collection.find():
                n += 1
                # print(cursor["id"])
                id = cursor["id"]
                self = cursor["self"]
                key = cursor["key"]

                if cursor.get("fields"):
                    if cursor["fields"].get("priority"):
                        priority = cursor["fields"]["priority"]["name"]

                    issuelinks = cursor["fields"]["issuelinks"]
                    if len(issuelinks) != 0:
                        m +=1
                    status = cursor["fields"]["status"]["name"]
                    issuetype = cursor["fields"]["issuetype"]["name"]
                    project = cursor["fields"]["project"]["name"]
                    summary = cursor["fields"]["summary"]

                    if cursor["fields"].get("description"):
                        description = cursor["fields"]["description"]
                    else:
                        description = summary
                    #
                    created = cursor["fields"]["created"]

                # _each_issue = {'id': id, 'self': self, 'key': key, 'priority': priority, 'issuelinks': issuelinks,
                #                'status': status, 'issuetype': issuetype, 'project': project, 'summary': summary,
                #                'description': description, 'created': created}

                _each_issue = {'id': id, 'self': self, 'key': key, 'priority': priority,
                               'status': status, 'issuetype': issuetype, 'project': project, 'summary': summary,
                               'description': description, 'created': created}
                file.write(json.dumps(_each_issue) + "\n")

                # _each_issue_links = {'id': id, 'key': key, 'issuelinks': issuelinks, 'created': created}
                #
                # file_issue_links.write(json.dumps(_each_issue_links) + "\n")
            print(_table, n, m, float(m/n))
            print("------------------------------------------\n")

    print("over")
    #



# _table :  JFrog
# JFrog 15535 3269 0.2104280656581912
# ------------------------------------------
#
# _table :  Apache
# Apache 1014926 210295 0.207202298492698
# ------------------------------------------
#
# _table :  Hyperledger
# Hyperledger 28146 6836 0.24287643004334541
# ------------------------------------------
#
# _table :  IntelDAOS
# IntelDAOS 9474 2648 0.27950179438463163
# ------------------------------------------
#
# _table :  Jira
# Jira 274545 127218 0.4633775883734907
# ------------------------------------------
#
# _table :  JiraEcosystem
# JiraEcosystem 41866 9354 0.2234271246357426
# ------------------------------------------
#
# _table :  MariaDB
# MariaDB 31229 12507 0.40049313138429027
# ------------------------------------------
#
# _table :  Mindville
# Mindville 2134 87 0.04076850984067479
# ------------------------------------------
#
# _table :  Mojang
# Mojang 420819 225918 0.5368531363840511
# ------------------------------------------
#
# _table :  MongoDB
# MongoDB 137172 56569 0.4123946578018838
# ------------------------------------------
#
# _table :  Qt
# Qt 148579 31588 0.2126007040025845
# ------------------------------------------
#
# _table :  RedHat
# RedHat 353000 116270 0.3293767705382436
# ------------------------------------------
#
# _table :  Sakai
# Sakai 50550 18838 0.37266073194856575
# ------------------------------------------
#
# _table :  SecondLife
# SecondLife 1867 464 0.2485270487412962
# ------------------------------------------
#
# _table :  Sonatype
# Sonatype 87284 4719 0.05406489161816599
# ------------------------------------------
#
# _table :  Spring
# Spring 69156 14181 0.20505812944646884
# ------------------------------------------
#
# over
#
# Process finished with exit code 0
