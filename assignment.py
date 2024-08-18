import pandas as pd
import random
from deap import tools, creator, base, algorithms
import numpy as np
import matplotlib.pyplot as plt

# Load and parse the CSV file
file_path = 'processed_modules_combined.csv'
df = pd.read_csv(file_path)

# Predefined required modules
required_modules_ai = df[df['PROGRAMME'] == 'AI'].sort_values(by='COUNT', ascending=False)
required_modules_ai = required_modules_ai.iloc[:-1]
required_modules_dsba = df[df['PROGRAMME'] == 'DSBA'].sort_values(by='COUNT', ascending=False)
required_modules_dsba = required_modules_dsba.iloc[:-1]

# Ensure required modules are a subset of all available modules
modules_ai = df[df['PROGRAMME'] == 'AI'].sort_values(by='COUNT', ascending=False)
modules_ai = modules_ai.iloc[:-1]
modules_dsba = df[df['PROGRAMME'] == 'DSBA'].sort_values(by='COUNT', ascending=False)
modules_dsba = modules_dsba.iloc[:-1]

required_modules_ai = [module for module in required_modules_ai['MODULE NAME'] if module in modules_ai['MODULE NAME']]
required_modules_dsba = [module for module in required_modules_dsba['MODULE NAME'] if module in modules_dsba['MODULE NAME']]

all_ai_modules = modules_ai['MODULE NAME'].tolist()
all_dsba_modules = modules_dsba['MODULE NAME'].tolist()

# Determine the top modules excluding RMCE and RMCP
top_modules_ai = modules_ai[~modules_ai['MODULE NAME'].isin(['RMCE'])].head(5)
top_modules_dsba = modules_dsba[~modules_dsba['MODULE NAME'].isin(['RMCP'])].head(5)

# Create the module preferences dictionary
module_preferences_ai = {}
module_preferences_dsba = {}

# AI Module Preferences
for i, module in enumerate(top_modules_ai['MODULE NAME']):
    if i < 1:
        module_preferences_ai[module] = 3  # Top 1 and 2 appear 3 times
    else:
        module_preferences_ai[module] = 2  # Top 3 and 4 appear 2 times

# DSBA Module Preferences
for i, module in enumerate(top_modules_dsba['MODULE NAME']):
    if i < 1:
        module_preferences_dsba[module] = 3  # Top 1 and 2 appear 3 times
    else:
        module_preferences_dsba[module] = 2  # Top 3 and 4 appear 2 times

# Ensure RMCE and RMCP appear 3 times
module_preferences_ai['RMCE'] = 3
module_preferences_dsba['RMCP'] = 3

# Ensure all other modules appear at least once
for module in all_ai_modules:
    if module not in module_preferences_ai:
        module_preferences_ai[module] = 1

for module in all_dsba_modules:
    if module not in module_preferences_dsba:
        module_preferences_dsba[module] = 1

intakes = ['Jan', 'Mar', 'May', 'Aug', 'Oct']

# Shared modules between AI and DSBA programs
shared_modules = ['BIS', 'MMDA', 'AML', 'DL', 'NLP']


class Schedule:
    def __init__(self, modules_ai, modules_dsba, intakes, required_modules_ai, required_modules_dsba, shared_modules,
                 hardConstraintPenalty, softConstraintPenalty, module_preferences_ai, module_preferences_dsba):
        self.modules_ai = modules_ai
        self.modules_dsba = modules_dsba
        self.intakes = intakes
        self.required_modules_ai = required_modules_ai
        self.required_modules_dsba = required_modules_dsba
        self.shared_modules = shared_modules
        self.hardConstraintPenalty = hardConstraintPenalty
        self.softConstraintPenalty = softConstraintPenalty
        self.module_preferences_ai = module_preferences_ai
        self.module_preferences_dsba = module_preferences_dsba

    def __len__(self):
        return len(self.intakes) * 4 * 2  # Ensure exactly 4 modules per intake for AI and DSBA

    def getCost(self, schedule):
        violations = 0

        len_ai = len(self.modules_ai)
        len_dsba = len(self.modules_dsba)

        schedule_ai = schedule[:len(self.intakes) * 4]
        schedule_dsba = schedule[len(self.intakes) * 4:]

        intake_modules_ai = {intake: [] for intake in self.intakes}
        intake_modules_dsba = {intake: [] for intake in self.intakes}

        for idx, module_idx in enumerate(schedule_ai):
            if module_idx < len_ai:
                module = self.modules_ai.iloc[module_idx]['MODULE NAME']
                intake = self.intakes[idx // 4]
                if module != 'None':
                    intake_modules_ai[intake].append(module)

        for idx, module_idx in enumerate(schedule_dsba):
            if module_idx < len_dsba:
                module = self.modules_dsba.iloc[module_idx]['MODULE NAME']
                intake = self.intakes[idx // 4]
                if module != 'None':
                    intake_modules_dsba[intake].append(module)

        # Constraint: Exactly 4 modules per intake
        exact_modules_per_intake_violations = 0
        for intake in self.intakes:
            if len(intake_modules_ai[intake]) != 4:
                exact_modules_per_intake_violations += abs(len(intake_modules_ai[intake]) - 4)
            if len(intake_modules_dsba[intake]) != 4:
                exact_modules_per_intake_violations += abs(len(intake_modules_dsba[intake]) - 4)
        violations += exact_modules_per_intake_violations * self.hardConstraintPenalty

        # Constraint: No consecutive module offerings within the same intake
        consecutive_module_violations = 0
        for intake in self.intakes:
            if len(intake_modules_ai[intake]) != len(set(intake_modules_ai[intake])):
                consecutive_module_violations += 1
            if len(intake_modules_dsba[intake]) != len(set(intake_modules_dsba[intake])):
                consecutive_module_violations += 1
        violations += consecutive_module_violations * self.hardConstraintPenalty

        # Constraint: No consecutive module offerings across intakes
        consecutive_module_across_intakes_violations = 0
        for i in range(1, len(self.intakes)):
            for module in self.modules_ai['MODULE NAME']:
                if module in intake_modules_ai[self.intakes[i]] and module in intake_modules_ai[self.intakes[i - 1]]:
                    consecutive_module_across_intakes_violations += 1
            for module in self.modules_dsba['MODULE NAME']:
                if module in intake_modules_dsba[self.intakes[i]] and module in intake_modules_dsba[self.intakes[i - 1]]:
                    consecutive_module_across_intakes_violations += 1
        violations += consecutive_module_across_intakes_violations * self.hardConstraintPenalty

        # Constraint: Shared module offerings
        shared_module_violations = 0
        for intake in self.intakes:
            for module in self.shared_modules:
                if module in intake_modules_ai[intake] and module not in intake_modules_dsba[intake]:
                    shared_module_violations += 1
                if module not in intake_modules_ai[intake] and module in intake_modules_dsba[intake]:
                    shared_module_violations += 1
        violations += shared_module_violations * self.hardConstraintPenalty

        # Constraint: RMCE and RMCP must be in Jan, May, Oct
        rm_module_violations = 0
        for intake in ['Jan', 'May', 'Oct']:
            if 'RMCE' not in intake_modules_ai[intake]:
                rm_module_violations += 1
            if 'RMCP' not in intake_modules_dsba[intake]:
                rm_module_violations += 1
        violations += rm_module_violations * self.hardConstraintPenalty * 10  # Increased penalty

        # Constraint: Ensure all required modules are offered at least once
        missing_required_module_violations = 0
        for module in self.required_modules_ai:
            if not any(module in intake_modules_ai[intake] for intake in self.intakes):
                missing_required_module_violations += 1
        for module in self.required_modules_dsba:
            if not any(module in intake_modules_dsba[intake] for intake in self.intakes):
                missing_required_module_violations += 1
        violations += missing_required_module_violations * self.softConstraintPenalty

        # Constraint: No duplicate module offerings within a single intake
        duplicate_module_violations = 0
        for intake in self.intakes:
            duplicate_module_violations += len(intake_modules_ai[intake]) - len(set(intake_modules_ai[intake]))
            duplicate_module_violations += len(intake_modules_dsba[intake]) - len(set(intake_modules_dsba[intake]))
        violations += duplicate_module_violations * self.hardConstraintPenalty

        # New Constraint: Module preferences frequency requirement
        module_preferences_violations = 0
        for module, required_frequency in self.module_preferences_ai.items():
            actual_frequency = sum(module in intake_modules_ai[intake] for intake in self.intakes)
            if actual_frequency < required_frequency:
                module_preferences_violations += (required_frequency - actual_frequency)

        for module, required_frequency in self.module_preferences_dsba.items():
            actual_frequency = sum(module in intake_modules_dsba[intake] for intake in self.intakes)
            if actual_frequency < required_frequency:
                module_preferences_violations += (required_frequency - actual_frequency)

        violations += module_preferences_violations * self.softConstraintPenalty

        return (
        violations, exact_modules_per_intake_violations, consecutive_module_violations, consecutive_module_across_intakes_violations, shared_module_violations,
        rm_module_violations, missing_required_module_violations, duplicate_module_violations,
        module_preferences_violations)

    def printScheduleInfo(self, schedule):
        len_ai = len(self.modules_ai)
        len_dsba = len(self.modules_dsba)

        schedule_ai = schedule[:len(self.intakes) * 4]
        schedule_dsba = schedule[len(self.intakes) * 4:]

        intake_modules_ai = {intake: [] for intake in self.intakes}
        intake_modules_dsba = {intake: [] for intake in self.intakes}

        for idx, module_idx in enumerate(schedule_ai):
            if module_idx < len_ai:
                module = self.modules_ai.iloc[module_idx]['MODULE NAME']
                intake = self.intakes[idx // 4]
                if module != 'None':
                    intake_modules_ai[intake].append(module)

        for idx, module_idx in enumerate(schedule_dsba):
            if module_idx < len_dsba:
                module = self.modules_dsba.iloc[module_idx]['MODULE NAME']
                intake = self.intakes[idx // 4]
                if module != 'None':
                    intake_modules_dsba[intake].append(module)

        print("Recommended Schedule for AI Programme:")
        for intake, modules in intake_modules_ai.items():
            print(f"{intake}: {', '.join(modules) if modules else 'No modules'}")

        print("\nRecommended Schedule for DSBA Programme:")
        for intake, modules in intake_modules_dsba.items():
            print(f"{intake}: {', '.join(modules) if modules else 'No modules'}")

        # Calculate violations
        (violations, exact_modules_per_intake_violations, consecutive_module_violations, consecutive_module_across_intakes_violations, shared_module_violations,
         rm_module_violations, missing_required_module_violations, duplicate_module_violations,
         module_preferences_violations) = self.getCost(schedule)

        print("\nViolations:")
        print(f"Exact Modules Per Intake Violations: {exact_modules_per_intake_violations}")
        print(f"Consecutive Module Violations (within same intake): {consecutive_module_violations}")
        print(f"Consecutive Module Violations (across intakes): {consecutive_module_across_intakes_violations}")
        print(f"Shared Module Violations: {shared_module_violations}")
        print(f"RM Module Violations: {rm_module_violations}")
        print(f"Missing Required Module Violations: {missing_required_module_violations}")
        print(f"Duplicate Module Violations: {duplicate_module_violations}")
        print(f"Module Preferences Violations: {module_preferences_violations}")
        print(f"Total Violations: {violations}")


# Setup GA parameters
populationSize = 600
generations = 100
pCrossover = 0.9
pMutation = 0.4

sch = Schedule(modules_ai, modules_dsba, intakes, required_modules_ai, required_modules_dsba, shared_modules, 100, 10,
               module_preferences_ai, module_preferences_dsba)

random.seed(42)

# Set up the toolbox
toolbox = base.Toolbox()
toolbox.register('gene', random.randint, 0, max(len(modules_ai), len(modules_dsba)) - 1)
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)
toolbox.register('individualCreator', tools.initRepeat, creator.Individual, toolbox.gene, len(sch))
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


# Set up the fitness function
def fitnessFunction(individual):
    return sch.getCost(individual)[0],


# Set up the genetic operators
toolbox.register('evaluate', fitnessFunction)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1 / len(sch))

# Define and run the algorithm
startingPop = toolbox.populationCreator(populationSize)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)

hof = tools.HallOfFame(5)

finalPop, logbook = algorithms.eaSimple(startingPop, toolbox, pCrossover, pMutation, generations, stats, hof, True)

minValue, avgValue = logbook.select('min', 'avg')

plt.plot(minValue, color='red')
plt.plot(avgValue, color='green')
plt.xlabel('generations')
plt.ylabel('min/avg fitness per generation')
plt.show()

best = hof[0]
sch.printScheduleInfo(best)
