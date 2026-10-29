"""One-off: append the Stage 3 held-out resumes and cases to labeled_pairs.json.

The labels below were written from what a hiring manager would say about each
requirement, before any threshold was fitted to them. The first 24 cases (three
resumes) are the development set the matcher was originally tuned on; the cases
added here are the held-out set.
"""
import json
import pathlib

PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labeled_pairs.json"
data = json.loads(PATH.read_text(encoding="utf-8"))

data["resumes"].update({
    "devops_fresher": (
        "PROJECTS\nCI Pipeline for Student Portal\n"
        "- Set up a Jenkins pipeline that builds, tests and deploys the portal on every merge, cutting release time from 2 hours to 15 minutes\n"
        "- Wrote Terraform modules to provision 3 EC2 instances and an S3 bucket\n"
        "- Configured Prometheus and Grafana dashboards tracking latency and error rate\n"
        "INTERNSHIP\nCloud Intern, Zeta Systems (Jun-Aug 2026)\n"
        "- Migrated 6 services to Docker containers on AWS ECS\n"
        "- Automated nightly database backups with a Bash script, saving 5 hours per week\n"
        "SKILLS\nLinux, Docker, Kubernetes, Ansible, Python, Git, Azure\n"
    ),
    "ml_fresher": (
        "PROJECTS\nSentiment Classifier for Product Reviews\n"
        "- Fine-tuned a DistilBERT model on 25k reviews reaching 91% accuracy\n"
        "- Built a preprocessing pipeline with NLTK and spaCy\n"
        "Crop Disease Detection\n"
        "- Trained a CNN in PyTorch on 12k leaf images, improving F1 from 0.71 to 0.84\n"
        "- Deployed the model behind a Flask API used in a campus demo\n"
        "INTERNSHIP\nData Science Intern, Orbit Analytics (May-Jul 2026)\n"
        "- Reduced churn model error by 12% through feature engineering in pandas and scikit-learn\n"
        "- Presented weekly findings to a team of 8\n"
        "SKILLS\nPython, SQL, PyTorch, TensorFlow, Tableau, Docker, Git\n"
    ),
    "design_fresher": (
        "PROJECTS\nCampus Bus Tracker Redesign\n"
        "- Redesigned the bus tracker app in Figma, cutting steps to find a route from 6 to 3\n"
        "- Ran usability tests with 14 students and fixed the 5 most reported issues\n"
        "- Built a component library of 30 reusable elements\n"
        "Portfolio Website\n"
        "- Developed a responsive portfolio in HTML, CSS and JavaScript\n"
        "INTERNSHIP\nUI Design Intern, Pixelcraft (Jun-Aug 2026)\n"
        "- Designed 12 mobile screens for a fintech onboarding flow\n"
        "- Worked with developers to ship the flow\n"
        "SKILLS\nFigma, Adobe XD, HTML, CSS, JavaScript, Photoshop, Illustrator\n"
    ),
    "fullstack_fresher": (
        "PROJECTS\nExpense Tracker\n"
        "- Built a full-stack app with React and Node.js serving 300 users\n"
        "- Designed a PostgreSQL schema with 8 tables and added indexes that cut dashboard load from 2s to 400ms\n"
        "- Implemented JWT authentication with refresh tokens\n"
        "- Wrote integration tests with Jest covering 70% of the API\n"
        "Chat Application\n"
        "- Added real-time messaging with WebSockets\n"
        "INTERNSHIP\nSoftware Intern, Brightlane (May-Jul 2026)\n"
        "- Fixed 20 bugs and shipped 3 features in a React codebase\n"
        "- Reviewed 15 pull requests on GitHub\n"
        "SKILLS\nJavaScript, TypeScript, React, Node.js, PostgreSQL, MongoDB, Docker, AWS, Git\n"
    ),
})


def c(resume, label, quote, expect, why):
    return {"resume": resume, "label": label, "quote": quote, "expect": expect, "why": why}


data["cases"] += [
    c("devops_fresher", "CI/CD pipelines", "Build and maintain CI/CD pipelines", "proven", "Jenkins pipeline on every merge, release time cut"),
    c("devops_fresher", "Infrastructure as code", "Provision infrastructure using Terraform or CloudFormation", "proven", "Terraform modules provisioning EC2 and S3"),
    c("devops_fresher", "Monitoring and alerting", "Set up monitoring and alerting with Prometheus and Grafana", "proven", "Dashboards for latency and error rate"),
    c("devops_fresher", "Docker containers", "Run containerised workloads with Docker", "proven", "Migrated 6 services to Docker on ECS"),
    c("devops_fresher", "AWS deployment", "Deploy and operate services on AWS", "proven", "EC2, S3 and ECS work"),
    c("devops_fresher", "Shell scripting", "Automate operational tasks with shell scripts", "proven", "Bash backup automation, hours saved"),
    c("devops_fresher", "Kubernetes orchestration", "Orchestrate containers with Kubernetes", "partial", "Kubernetes only in the skills list"),
    c("devops_fresher", "Configuration management", "Manage configuration with Ansible", "partial", "Ansible listed, never used"),
    c("devops_fresher", "Azure cloud services", "Work with Azure cloud services", "partial", "Azure listed only"),
    c("devops_fresher", "Java backend", "Develop backend services in Java", "missing", "No Java anywhere"),
    c("devops_fresher", "Security scanning", "Implement vulnerability scanning and security compliance", "missing", "No security work"),
    c("devops_fresher", "Kafka operations", "Operate Kafka clusters", "missing", "No streaming work"),
    c("devops_fresher", "React interfaces", "Build user interfaces in React", "missing", "No frontend work"),

    c("ml_fresher", "Natural language processing", "Build natural language processing models", "proven", "Fine-tuned DistilBERT, NLTK and spaCy pipeline"),
    c("ml_fresher", "Deep learning", "Train deep learning models in PyTorch or TensorFlow", "proven", "CNN trained in PyTorch with F1 gain"),
    c("ml_fresher", "Feature engineering", "Perform feature engineering and evaluate model error", "proven", "Churn error reduced 12% via feature engineering"),
    c("ml_fresher", "Model deployment", "Deploy trained models behind an API", "proven", "Flask API serving the CNN"),
    c("ml_fresher", "Stakeholder communication", "Present findings to the team", "proven", "Weekly presentations to a team of 8"),
    c("ml_fresher", "SQL querying", "Query large datasets using SQL", "partial", "SQL listed, no queries described"),
    c("ml_fresher", "Tableau dashboards", "Build dashboards in Tableau or Power BI", "partial", "Tableau listed only"),
    c("ml_fresher", "Docker for ML", "Containerise machine learning services with Docker", "partial", "Docker listed only"),
    c("ml_fresher", "ML pipeline orchestration", "Operate machine learning pipelines with Airflow and MLflow", "missing", "No orchestration"),
    c("ml_fresher", "Big data processing", "Process large datasets with Spark", "missing", "No Spark or distributed work"),
    c("ml_fresher", "Kubernetes deployment", "Deploy services on Kubernetes", "missing", "No Kubernetes"),
    c("ml_fresher", "A/B testing", "Design and analyse A/B experiments", "missing", "No experimentation work"),

    c("design_fresher", "Prototyping in Figma", "Create wireframes and prototypes in Figma", "proven", "Redesigned the tracker in Figma, steps cut 6 to 3"),
    c("design_fresher", "Usability testing", "Conduct user research and usability testing", "proven", "14 students tested, issues fixed"),
    c("design_fresher", "Design systems", "Create and maintain design systems and component libraries", "proven", "Built a 30-element component library"),
    c("design_fresher", "Responsive front end", "Implement responsive layouts in HTML and CSS", "proven", "Responsive portfolio in HTML, CSS, JavaScript"),
    c("design_fresher", "Mobile interface design", "Design mobile app interfaces", "proven", "12 mobile screens for onboarding"),
    c("design_fresher", "Working with engineers", "Collaborate with engineers to ship designs", "proven", "Worked with developers to ship the flow"),
    c("design_fresher", "Adobe XD prototyping", "Prototype interactions in Adobe XD", "partial", "Adobe XD listed only"),
    c("design_fresher", "Illustration assets", "Create visual assets in Illustrator and Photoshop", "partial", "Tools listed, no asset work described"),
    c("design_fresher", "Accessibility", "Apply WCAG accessibility standards", "missing", "No accessibility work"),
    c("design_fresher", "Motion design", "Create motion graphics in After Effects", "missing", "No motion work"),
    c("design_fresher", "Product analytics", "Use product analytics to guide design decisions", "missing", "No analytics"),
    c("design_fresher", "Python scripting", "Write Python scripts for data work", "missing", "No Python"),

    c("fullstack_fresher", "React development", "Build and ship features in React", "proven", "Full-stack React app, features shipped in React codebase"),
    c("fullstack_fresher", "Node.js APIs", "Design and build REST APIs in Node.js", "proven", "Full-stack app on Node.js serving 300 users"),
    c("fullstack_fresher", "Relational database design", "Design relational database schemas and optimise queries", "proven", "PostgreSQL schema, indexes cut load time"),
    c("fullstack_fresher", "Authentication", "Implement user authentication and token handling", "proven", "JWT auth with refresh tokens"),
    c("fullstack_fresher", "Automated testing", "Write integration tests with meaningful coverage", "proven", "Jest tests covering 70% of the API"),
    c("fullstack_fresher", "Code review", "Participate in code review on pull requests", "proven", "Reviewed 15 pull requests"),
    c("fullstack_fresher", "Real-time features", "Build real-time features such as WebSockets", "proven", "Real-time messaging with WebSockets"),
    c("fullstack_fresher", "Docker", "Containerise applications with Docker", "partial", "Docker listed only"),
    c("fullstack_fresher", "AWS", "Deploy applications to AWS", "partial", "AWS listed only"),
    c("fullstack_fresher", "MongoDB", "Model data in MongoDB", "partial", "MongoDB listed only"),
    c("fullstack_fresher", "TypeScript", "Write production TypeScript", "partial", "TypeScript listed only"),
    c("fullstack_fresher", "Kubernetes", "Deploy workloads on Kubernetes", "missing", "No Kubernetes"),
    c("fullstack_fresher", "Machine learning features", "Apply machine learning to product features", "missing", "No ML"),
    c("fullstack_fresher", "Android development", "Build native Android applications", "missing", "No mobile"),
    c("fullstack_fresher", "GraphQL", "Design GraphQL APIs", "missing", "No GraphQL"),
]

data["_comment"] = (
    "Hand-labeled resume/requirement pairs. Each case states what a hiring manager would say "
    "about that requirement given that resume. The first 24 cases (backend, frontend and data "
    "resumes) are the development set the thresholds were originally fitted to. The 53 cases "
    "on the devops, ml, design and fullstack resumes were written before any threshold was "
    "fitted to them and form the held-out set. All labels are from a single annotator."
)
data["_dev_resumes"] = ["backend_fresher", "frontend_fresher", "data_fresher"]
PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(len(data["resumes"]), "resumes,", len(data["cases"]), "cases")
