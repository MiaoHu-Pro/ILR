import os

import numpy as np


def write_issue_issue_links(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) + '\t' + str(data[k][3]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_issue_links_with_time(rank_path):
    f = open(rank_path)
    f.readline()
    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()

    return np.array(x_obj)


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(file)
    return L


# file_name_list = file_name(issuesFilePath)

if __name__ == "__main__":

    relation_categories = {'general relation': ['Reference','Relationship' ,'Polaris issue link','Polaris datapoint issue link', 'Related','Relates', 'Relate', 'Related','1 - Relate'],
                           'duplication': ['Duplicate', 'Replacement', 'Cloners', 'Cloners', 'Cloners (old)','Cloners (migrated)',
                                           'Cloners','2 - Cloned', 'Clones', 'Cloned', 'Cloner', 'Clones', 'Cloned', '3 - Duplicate'],
                           'temporal causal': ['Blocker', 'Preceded By', '6 - Blocks', 'Gantt: start-finish', 'Gantt: start-start',
                                               'start-finish [GANTT]', 'Gantt: finish-start','Gantt End to Start', 'Gantt Start to Finish',
                                               'finish-start [GANTT]', 'finish-finish [GANTT]', 'Gantt: finish-finish',
                                               'Account', 'Causality', 'Work Breakdown', 'Collection',
                                               'Gantt End to End', 'Sequence', 'Gantt Start to Start', 'Blocks',
                                               'Follows', 'Required', 'Cause', 'Caused', '5 - Depend','Depends', 'Problem/Incident',
                                               'Gantt Dependency', 'dependent', 'Depend', 'Dependency', 'Dependent', 'Blocked', 'Finish-to-Finish link (WBSGantt)'],

                           'composition': ['Parent Feature', 'multi-level hierarchy [GANTT]', 'Initiative', 'Epic','Superset',
                                           'Incorporates', 'Part', 'PartOf','Child-Issue', 'Parent/Child', 'Subtask', '4 - Incorporate',
                                           'Container', 'Split', 'Issue split', 'Detail', 'Covered','Contains(WBSGantt)', 'Contains',],

                           'workflow': ['Backports', 'Documentation', 'Supercedes','Supersede', 'Tested', 'Documented', 'Resolve',
                                        'Supersession', 'Derived', 'Completes', 'Regression',
                                        'Bonfire Testing','Test', 'Bonfire testing', 'Testing', '7 - Git Code Review', 'Git Code Review','Implements' ,
                                        'Fixes', 'Fix', 'Fixes', 'Fixes','Implement', 'Verify','Trigger', 'Triggered', 'Triggering', 'Triggered', 'Triggering',]
                           }

    relation_unification = {}

    for key, values_list in relation_categories.items():
        for value in values_list:
            relation_unification[value] = key

    print(relation_unification)

    file_name = file_name("./issue_links")
    for file in file_name:
        triples = []
        print(file)
        path = "./issue_links/" + file
        _issue_links = read_issue_links_with_time(path)

        for _triplet in _issue_links:

            _triplet[1] = relation_unification[_triplet[1]]

            triples.append(_triplet)
        write_issue_issue_links("./issue_links_unification/" + file.split(".")[0].split("_")[0] + "_issue_links_unification.txt", triples)
