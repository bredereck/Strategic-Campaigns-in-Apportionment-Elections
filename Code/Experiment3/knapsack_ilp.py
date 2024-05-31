#!/usr/bin/python
from gurobipy import *

#itemsIn: dict of items partitioned into their category: categoryNumber: [(value, size)]
#budgetIn: knapsack max size
def solve_for_given_budget(itemsIn, budgetIn):

  m = Model("knapsack")
  m.setParam('OutputFlag', False)
  m.setParam('MIPGap', 0.0)

#  # disable presolving
#  m.setParam( 'Presolve', False)

  itemsCategoryTupleLists = []
  for entry in itemsIn.items():
   itemsCategoryTupleLists += [(entry[0], indexAndItem[0], indexAndItem[1]) for
       indexAndItem in enumerate(entry[1])]

  # create variables for each item
  items = m.addVars([(item[0], item[1]) for item in itemsCategoryTupleLists],
      vtype = GRB.BINARY, name = "item")

  #size constraint
  m.addConstr(quicksum([items[item[0], item[1]]*item[2][1] for item in
    itemsCategoryTupleLists])<=budgetIn)
  
  #category constraints
  for categoryNr in itemsIn.keys():
    m.addConstr(quicksum([items[item[0], item[1]] for item in itemsCategoryTupleLists
      if item[0]==categoryNr])<=1)

  #optimization objective
  m.setObjective(quicksum([items[item[0], item[1]]*item[2][0] for item in
    itemsCategoryTupleLists]), GRB.MAXIMIZE)

  m.optimize()

  # check if model is infeasible
  if m.status != GRB.Status.OPTIMAL:
    return None
    
  return int(m.objVal)


#itemsIn: dict of items partitioned into their category: categoryNumber: [(value, size)]
#budgetIn: knapsack max size
def solve_for_given_value(itemsIn, valueToReach):

  m = Model("knapsack")
  m.setParam('OutputFlag', False)
  m.setParam('MIPGap', 0.0)

#  # disable presolving
#  m.setParam( 'Presolve', False)

  itemsCategoryTupleLists = []
  for entry in itemsIn.items():
   itemsCategoryTupleLists += [(entry[0], indexAndItem[0], indexAndItem[1]) for
       indexAndItem in enumerate(entry[1])]

  # create variables for each item
  items = m.addVars([(item[0], item[1]) for item in itemsCategoryTupleLists],
      vtype = GRB.BINARY, name = "item")

  #value-to-reach constraint
  m.addConstr(quicksum([items[item[0], item[1]]*item[2][0] for item in
    itemsCategoryTupleLists])==valueToReach)
  
  #category constraints
  for categoryNr in itemsIn.keys():
    m.addConstr(quicksum([items[item[0], item[1]] for item in itemsCategoryTupleLists
      if item[0]==categoryNr])<=1)

  #optimization objective---minimize the size (price)
  m.setObjective(quicksum([items[item[0], item[1]]*item[2][1] for item in
    itemsCategoryTupleLists]), GRB.MINIMIZE)

  m.optimize()

  # check if model is infeasible
  if m.status != GRB.Status.OPTIMAL:
    return None
    
  return int(m.objVal)
