import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Developer:
    def __init__(self, id, name, experience, fix_time, success_rate, workload, domain_skill, on_leave=False):
        self.id = id
        self.name = name
        self.experience = experience  # Benefit (Years)
        self.fix_time = fix_time      # Cost (Average hours)
        self.success_rate = success_rate  # Benefit (Percentage)
        self.workload = workload      # Cost (Percentage 0-100)
        self.domain_skill = domain_skill  # Benefit (Match rate percentage)
        self.on_leave = on_leave      # Boolean constraint

class Bug:
    def __init__(self, id, description, severity, module):
        self.id = id
        self.description = description
        self.severity = severity
        self.module = module
        self.vector = None

class BugAssigner:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.historical_bugs = []
        self.historical_bug_vectors = None
        self.developers = []
        # Mapping bug_id -> developer_id who resolved it
        self.resolution_history = {}

    def fit_historical_data(self, bugs, developers, resolutions):
        self.historical_bugs = bugs
        self.developers = developers
        self.resolution_history = resolutions
        
        corpus = [bug.description for bug in self.historical_bugs]
        self.historical_bug_vectors = self.vectorizer.fit_transform(corpus).toarray()

    def get_dynamic_weights(self, bug):
        # Weight vector: [Experience, Fix Time, Success Rate, Workload, Domain Skill]
        # Sum of weights should be 1
        if bug.severity.lower() == 'critical':
            # Emphasize Fix Time (w2) and Domain Skill (w5)
            weights = np.array([0.1, 0.4, 0.1, 0.1, 0.3])
        elif bug.severity.lower() == 'minor':
            # Emphasize low Workload (w4)
            weights = np.array([0.1, 0.1, 0.1, 0.6, 0.1])
        else:
            # Default balanced weights
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return weights

    def assign_bug(self, new_bug, k=3):
        # 1. Bug Reporting & Feature Extraction
        new_bug.vector = self.vectorizer.transform([new_bug.description]).toarray()[0]

        # 2. Historical Similarity Search (KNN)
        similarities = cosine_similarity([new_bug.vector], self.historical_bug_vectors)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # 3. Candidate Identification and Hard Filtering
        candidate_dev_ids = set()
        for idx in top_k_indices:
            hist_bug = self.historical_bugs[idx]
            if hist_bug.id in self.resolution_history:
                candidate_dev_ids.add(self.resolution_history[hist_bug.id])

        candidate_devs = [dev for dev in self.developers if dev.id in candidate_dev_ids]
        
        # Hard filtering: Availability and Workload Threshold
        filtered_devs = [dev for dev in candidate_devs if not dev.on_leave and dev.workload <= 95]

        if not filtered_devs:
            # Fallback if all candidates are filtered out or k=0
            filtered_devs = [dev for dev in self.developers if not dev.on_leave and dev.workload <= 95]
            if not filtered_devs:
                return None, "No available developers"

        # 4. Attribute Matrix Generation
        # Attributes: [Experience, Fix Time, Success Rate, Workload, Domain Skill]
        attr_matrix = np.array([
            [dev.experience, dev.fix_time, dev.success_rate, dev.workload, dev.domain_skill]
            for dev in filtered_devs
        ])

        # 5. Dynamic Contextual AI Weighting
        weights = self.get_dynamic_weights(new_bug)

        # 6. Normalization and WSM Calculation
        # Min and Max for each criterion across current candidates
        mins = attr_matrix.min(axis=0)
        maxs = attr_matrix.max(axis=0)

        normalized_matrix = np.zeros_like(attr_matrix, dtype=float)
        
        for i in range(len(filtered_devs)):
            for j in range(5):
                # Handle division by zero if all values are the same
                if maxs[j] == mins[j]:
                    normalized_matrix[i, j] = 1.0 # Give full score if everyone is equal
                    continue
                
                # Benefit criteria: Experience (0), Success Rate (2), Domain Skill (4)
                if j in [0, 2, 4]:
                    normalized_matrix[i, j] = (attr_matrix[i, j] - mins[j]) / (maxs[j] - mins[j])
                # Cost criteria: Fix Time (1), Workload (3)
                else:
                    normalized_matrix[i, j] = (maxs[j] - attr_matrix[i, j]) / (maxs[j] - mins[j])

        # Calculate Utility Scores
        utility_scores = np.dot(normalized_matrix, weights)

        # 7. Developer Ranking and Final Assignment
        ranked_indices = utility_scores.argsort()[::-1]
        
        ranked_devs = [(filtered_devs[idx], utility_scores[idx]) for idx in ranked_indices]
        best_dev, best_score = ranked_devs[0]

        return best_dev, ranked_devs

# Example Usage
if __name__ == "__main__":
    # Mock Historical Data
    developers = [
        Developer(1, "Alice Chen", experience=8, fix_time=1.5, success_rate=98, workload=85, domain_skill=95),
        Developer(2, "Bob Smith", experience=4, fix_time=4.0, success_rate=82, workload=30, domain_skill=70),
        Developer(3, "Charlie Davis", experience=2, fix_time=6.5, success_rate=75, workload=10, domain_skill=55),
        Developer(4, "Diana Prince", experience=6, fix_time=2.2, success_rate=92, workload=45, domain_skill=88),
        Developer(5, "Evan Wright", experience=9, fix_time=1.8, success_rate=95, workload=98, domain_skill=92) # Workload > 95
    ]

    historical_bugs = [
        Bug(101, "App crashes when uploading large image >5MB", "critical", "media"),
        Bug(102, "CSS styling issue on the login button", "minor", "ui"),
        Bug(103, "Database connection timeout during peak hours", "critical", "backend"),
        Bug(104, "Typo in the settings menu text", "minor", "ui"),
        Bug(105, "Memory leak in the background worker process", "critical", "backend")
    ]

    resolutions = {
        101: 4, # Diana Prince solved the image crash
        102: 2, # Bob Smith solved the UI bug
        103: 1, # Alice solved the DB timeout
        104: 3, # Charlie solved the typo
        105: 1  # Alice solved the memory leak
    }

    assigner = BugAssigner()
    assigner.fit_historical_data(historical_bugs, developers, resolutions)

    print("--- Testing Critical Bug Assignment ---")
    new_bug_critical = Bug(201, "App completely crashes when trying to upload video file", "critical", "media")
    best_dev, rankings = assigner.assign_bug(new_bug_critical, k=2)
    
    if best_dev:
        print(f"Optimal Assignment: {best_dev.name} (Score: {rankings[0][1]:.3f})")
        print("Candidate Rankings:")
        for dev, score in rankings:
            print(f" - {dev.name}: {score:.3f}")
    else:
        print("No assignment made.", rankings)
        
    print("\n--- Testing Minor Bug Assignment ---")
    new_bug_minor = Bug(202, "Alignment issue on the profile page avatar", "minor", "ui")
    best_dev, rankings = assigner.assign_bug(new_bug_minor, k=2)
    
    if best_dev:
        print(f"Optimal Assignment: {best_dev.name} (Score: {rankings[0][1]:.3f})")
        print("Candidate Rankings:")
        for dev, score in rankings:
            print(f" - {dev.name}: {score:.3f}")
    else:
        print("No assignment made.", rankings)
