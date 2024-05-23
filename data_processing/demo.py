import random



Task: {
    general relation:{  Technical task, Test, Task, Brainstorming, Question, Epic,\\
                        Improvement, Story, Suitable Name Search, New JIRA Project,\\
                        Github Integration, Documentation, Dependency upgrade,\\
                        Migration, New TLP , Project, Umbrella, Wish, Bug,\\
                        New Git Repo, New Feature, Planned Work, Sub-task},
    duplication: {   Wish, Technical task, New Feature, Test, Task,
                     Suitable Name Search, Story, Epic, Github Integration,
                     Documentation, Dependency upgrade, New TLP , Planned Work,
                     Bug, Sub-task, Improvement},
workflow: {       Wish, Technical task, Story, Test, Task, Github Integration,
                  Epic, Sub-task, Dependency upgrade, Documentation, Question,
                  Bug, New Feature, Improvement, New Git Repo},
    composition: {Wish, Story, Test, Task, New JIRA Project, Epic, Sub-task,
             Dependency upgrade, Question, Planned Work, Bug, New Feature,
             Improvement, Umbrella},
    temporal causal: {Technical task, Test, Task, Question, Epic, Improvement,
                 Story, GitBox Request, Suitable Name Search, New JIRA Project,
                 Github Integration, Documentation, Dependency upgrade,
                 Migration, New TLP , Project, Umbrella, Wish, Bug,
                 New Git Repo, New Feature, Planned Work, Mirroring,
                 Sub-task, Request}}










if __name__ ==  __main__ :

    for i in range(6):
        num = 0
        list_ = []
        for j in range(100):
            list_.append(random.randint(0, 10000000))
            num += 1
            if num >= 10:
                break



        print(list_)

        s = random.sample([0, 6, 5, 4, 3, 4, 5, 6, 6], 4)
        print(s)
