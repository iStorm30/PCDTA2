package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
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
	rand.Seed(time.Now().UnixNano())

	// Generar datos de ejemplo
	numExamples := 100000
	examples := make([]Example, numExamples)
	for i := 0; i < numExamples; i++ {
		features := make([]float64, 4) // 4 features para IRIS dataset
		for j := range features {
			features[j] = rand.Float64() * 10 // Números aleatorios entre 0 y 10
		}
		class := "ClassA"
		if i%2 == 0 {
			class = "ClassB"
		}
		examples[i] = Example{
			Features: features,
			Class:    class,
		}
	}

	// Medir tiempo de entrenamiento
	startTime := time.Now()

	// Construir árbol de decisión concurrentemente
	tree := BuildDecisionTreeConcurrent(examples, 0)

	// Medir tiempo después del entrenamiento
	elapsed := time.Since(startTime)

	// Imprimir el árbol de decisión
	PrintDecisionTree(tree, 0)

	// Imprimir tiempo de entrenamiento
	fmt.Println("Tiempo de entrenamiento:", elapsed)
}

func BuildDecisionTreeConcurrent(examples []Example, depth int) *DecisionTree {
	// Si no hay ejemplos o se alcanza la profundidad máxima, devuelve un nodo hoja con la clase mayoritaria
	if len(examples) == 0 || depth >= 3 {
		return &DecisionTree{
			Class: MajorityClass(examples),
		}
	}

	// Encontrar la mejor división de forma concurrente
	bestSplit := FindBestSplitConcurrent(examples)

	// Si no se encuentra la mejor división, devuelve un nodo hoja con la clase mayoritaria
	if bestSplit == nil {
		return &DecisionTree{
			Class: MajorityClass(examples),
		}
	}

	// Dividir ejemplos
	var leftExamples, rightExamples []Example
	for _, example := range examples {
		if example.Features[bestSplit.Column] <= bestSplit.Value {
			leftExamples = append(leftExamples, example)
		} else {
			rightExamples = append(rightExamples, example)
		}
	}

	// Construir recursivamente los subárboles izquierdo y derecho de forma concurrente
	var wg sync.WaitGroup
	wg.Add(2)

	var left *DecisionTree
	var right *DecisionTree

	go func() {
		left = BuildDecisionTreeConcurrent(leftExamples, depth+1)
		wg.Done()
	}()

	go func() {
		right = BuildDecisionTreeConcurrent(rightExamples, depth+1)
		wg.Done()
	}()

	wg.Wait()

	return &DecisionTree{
		Left:   left,
		Right:  right,
		Column: bestSplit.Column,
		Value:  bestSplit.Value,
	}
}

func FindBestSplitConcurrent(examples []Example) *DecisionTree {
	if len(examples) == 0 {
		return nil
	}

	numFeatures := len(examples[0].Features)
	bestGini := math.Inf(1)
	var bestSplit *DecisionTree

	type SplitResult struct {
		Split *DecisionTree
		Gini  float64
	}

	results := make(chan SplitResult)

	for col := 0; col < numFeatures; col++ {
		go func(col int) {
			// Ordenar ejemplos por valor de característica
			sort.Slice(examples, func(i, j int) bool {
				return examples[i].Features[col] < examples[j].Features[col]
			})

			for i := 1; i < len(examples); i++ {
				// Probar división en punto medio
				value := (examples[i-1].Features[col] + examples[i].Features[col]) / 2.0

				// Dividir ejemplos
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

				// Calcular impureza de Gini
				gini := CalculateGini(leftClasses, rightClasses, leftCount, rightCount)

				// Actualizar mejor división si es mejor
				if gini < bestGini {
					bestGini = gini
					bestSplit = &DecisionTree{
						Column: col,
						Value:  value,
					}
				}
			}

			results <- SplitResult{Split: bestSplit, Gini: bestGini}
		}(col)
	}

	// Obtener resultado de la goroutine más rápida
	for i := 0; i < numFeatures; i++ {
		result := <-results
		if result.Gini < bestGini {
			bestGini = result.Gini
			bestSplit = result.Split
		}
	}

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
