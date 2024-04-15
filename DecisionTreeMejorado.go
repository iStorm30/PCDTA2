package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
)

type DecisionTree struct {
	Left   *DecisionTree
	Right  *DecisionTree
	Column int
	Value  float64
	Class  string
}

type Example struct {
	Features []float64
	Class    string
}

func main() {
	// Load CSV data
	data, err := LoadCSV("IRIS.csv")
	if err != nil {
		log.Fatal(err)
	}

	// Convert data to examples
	examples := make([]Example, len(data))
	for i, d := range data {
		features := make([]float64, len(d)-1)
		for j := range features {
			features[j], _ = strconv.ParseFloat(d[j], 64)
		}
		examples[i] = Example{
			Features: features,
			Class:    d[len(d)-1],
		}
	}

	// Build decision tree
	tree := BuildDecisionTree(examples, 0)

	// Print the decision tree
	PrintDecisionTree(tree, 0)
}

func LoadCSV(filename string) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return data, nil
}

func BuildDecisionTree(examples []Example, depth int) *DecisionTree {
	// If no examples or max depth reached, return a leaf node with the majority class
	if len(examples) == 0 || depth >= 3 {
		return &DecisionTree{
			Class: MajorityClass(examples),
		}
	}

	// Find the best split
	bestSplit := FindBestSplit(examples)

	// If no best split found, return a leaf node with the majority class
	if bestSplit == nil {
		return &DecisionTree{
			Class: MajorityClass(examples),
		}
	}

	// Split examples
	var leftExamples, rightExamples []Example
	for _, example := range examples {
		if example.Features[bestSplit.Column] <= bestSplit.Value {
			leftExamples = append(leftExamples, example)
		} else {
			rightExamples = append(rightExamples, example)
		}
	}

	// Recursively build left and right subtrees
	left := BuildDecisionTree(leftExamples, depth+1)
	right := BuildDecisionTree(rightExamples, depth+1)

	return &DecisionTree{
		Left:   left,
		Right:  right,
		Column: bestSplit.Column,
		Value:  bestSplit.Value,
	}
}

func FindBestSplit(examples []Example) *DecisionTree {
	if len(examples) == 0 {
		return nil
	}

	numFeatures := len(examples[0].Features)
	bestGini := math.Inf(1)
	var bestSplit *DecisionTree

	var wg sync.WaitGroup
	wg.Add(numFeatures)

	for col := 0; col < numFeatures; col++ {
		go func(col int) {
			defer wg.Done()

			// Sort examples by feature value
			sort.Slice(examples, func(i, j int) bool {
				return examples[i].Features[col] < examples[j].Features[col]
			})

			for i := 1; i < len(examples); i++ {
				// Try splitting at midpoint
				value := (examples[i-1].Features[col] + examples[i].Features[col]) / 2.0

				// Split examples
				var leftCount, rightCount int
				var leftClasses, rightClasses map[string]int
				leftClasses = make(map[string]int)
				rightClasses = make(map[string]int)

				for _, example := range examples {
					if example.Features[col] <= value {
						leftCount++
						leftClasses[example.Class]++
					} else {
						rightCount++
						rightClasses[example.Class]++
					}
				}

				// Calculate Gini impurity
				gini := CalculateGini(leftClasses, rightClasses, leftCount, rightCount)

				// Update best split if this is better
				if gini < bestGini {
					bestGini = gini
					bestSplit = &DecisionTree{
						Column: col,
						Value:  value,
					}
				}
			}
		}(col)
	}

	wg.Wait()

	return bestSplit
}

func CalculateGini(leftClasses, rightClasses map[string]int, leftCount, rightCount int) float64 {
	total := float64(leftCount + rightCount)
	giniLeft := GiniImpurity(leftClasses, leftCount)
	giniRight := GiniImpurity(rightClasses, rightCount)
	gini := (float64(leftCount)/total)*giniLeft + (float64(rightCount)/total)*giniRight
	return gini
}

func GiniImpurity(classCounts map[string]int, totalCount int) float64 {
	if totalCount == 0 {
		return 0.0
	}

	var impurity float64
	for _, count := range classCounts {
		prob := float64(count) / float64(totalCount)
		impurity += prob * (1 - prob)
	}

	return impurity
}

func MajorityClass(examples []Example) string {
	classCounts := make(map[string]int)
	for _, example := range examples {
		classCounts[example.Class]++
	}

	maxCount := 0
	var majorityClass string
	for class, count := range classCounts {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}

	return majorityClass
}

func PrintDecisionTree(tree *DecisionTree, indent int) {
	if tree == nil {
		return
	}

	for i := 0; i < indent; i++ {
		fmt.Print("  ")
	}

	if tree.Left == nil && tree.Right == nil {
		fmt.Println("Class:", tree.Class)
	} else {
		fmt.Printf("Feature %d <= %.2f\n", tree.Column, tree.Value)
		PrintDecisionTree(tree.Left, indent+1)

		for i := 0; i < indent; i++ {
			fmt.Print("  ")
		}
		fmt.Println("else")
		PrintDecisionTree(tree.Right, indent+1)
	}
}
