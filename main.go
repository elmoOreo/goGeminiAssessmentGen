package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"os"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type empty struct{}

type assessmentValidatedData struct {
	Question           string
	ValidatedAnswer    string
	ValidatedReasoning string
}

type assessmentData struct {
	Subject     string
	Topic       string
	Proficiency string
	Question    string
	Answer      string
	AllOptions  []string
	Reasoning   string
	Complexity  string
	Source      string
	LLMName     string
}

type assessmentDataforMap struct {
	Subject              string
	Topic                string
	Proficiency          string
	Question             string
	Answer               string
	AllOptions           []string
	Reasoning            string
	Complexity           string
	Source               string
	LLMName              string
	ValidatedAnswer      string
	ValidatedReasoning   string
	ValidatedSelectedLLM string
}

var systemPrompt = `You are an AI Guru and an expert in AI literature. You are tasked to generate a set of multiple choice assessments 
for evaluating a Talent based on their proficiency on multiple Topics in a particular Subject area.
The talent can belong to one of the following Proficiencies in the increasing order of expertise either a Learner, Practitioner or Specialist. 
The following is the definition of each Proficiency
1) Learner      : A Learner is at a level 0 or foundation level. A Learner is aware of the basic concepts and has little or minimal practical experience.
2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable and has an advanced understanding of the concept and has meaningful practical experience of implementing the concept over a few years.
3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has meaningful practical experience of implementing the concept over many years.

The assessments against each proficiency on a topic will have a mix of Easy, Medium or Difficult questions designed to test the Talent. 
The following is the definition of each level of Complexity
1) Easy     	: Easy questions fall around basic knowledge or understanding. There is no ambiguity or hidden meanings. It often involves a single calculation or task.
2) Medium 		: Medium questions demand a deeper understanding of the subject matter. It involes reasoning.
3) Difficult	: Difficult questions centre around combination different ideas or principles. It demands analysis, evaluation, and synthesis of information.

Please do not hallucinate, if you are not aware, please say it so in courteous fashion. 
Please do not share anything that can be construed as harmful.`

var systemPromptForValidation = `You are an expert in AI literature. Answer the following questions to the best of your knowledge. 
You will be prompted with a set of questions and a set of options for each question, choose only one of the right options for each question 
that accurately reflects the ask and also articulate why its the right answer. If you do not know the answer to any question, 
please say I do not know. If the right accurate option for the question does not exist, please say The right option is not listed`

func getPromptRefinedforValidation(allQuizes []assessmentDataforMap) string {

	var promptTemplate string

	var stepsSequencePrompt string

	stepsPrompt := `

	The following are the Questions and the Options

	Question %d:
	%s

	Options:
	%s
	%s
	%s
	%s
	I do not know
	The right option is not listed`

	for outIdx := 0; outIdx < len(allQuizes); outIdx++ {
		stepsPromptInit := fmt.Sprintf(stepsPrompt, outIdx, allQuizes[outIdx].Question, allQuizes[outIdx].AllOptions[0],
			allQuizes[outIdx].AllOptions[1], allQuizes[outIdx].AllOptions[2], allQuizes[outIdx].AllOptions[3])
		stepsSequencePrompt = stepsSequencePrompt + stepsPromptInit
	}

	outputforPrompt := `
	
	The following will be part of the results
	1) Question as Question
	2) Answer as ValidatedAnswer
	3) Reasoning as ValidatedReasoning

	Return the results using this JSON schema:
	ValidatedAssessment = {
	'Question' : string
	'ValidatedAnswer': string
	'ValidatedReasoning': string
	}
	Return: Array<ValidatedAssessment>`

	promptTemplate = fmt.Sprintf("\n  %s \n %s", stepsSequencePrompt, outputforPrompt)

	return promptTemplate

}

func getPromptRefined(assessmentBankCount int, proficiency string, complexity string, topic string, subTopic string, llmName string) string {

	stepsPrompt := `You will build an assessment bank of atleast %d questions of %s Complexity for evaluating the Proficiency of a %s on the Subject of %s

	The following will be the steps to generate the assessment bank

	Step 1) Ask a relevant Question on the Topic of %s within the context of %s.
	Step 2) If the Question is a repeat, ask a different relevant Question
	Step 3) For each Question generated, List the Answer having a maximum length of no more than 5 words to the Question.
	Step 4) For each Question generated, Generate 3 other similar answers having a maximum length of no more than 5 words.
	Step 5) For each Question generated, Return the answers generated in Step 3 and Step 4 as a single list as AllOptions.
	Step 6) For each Question generated, Articulate in detail why the Answer in Step 3 is the right Answer for the Question as Reasoning.
	Step 7) For each Question generated, Estimate the Complexity of the question in terms of Easy, Medium or Difficult.
	Step 8) For each Question generated, Highlight the Source if any from which the question was articulated, do not hallucinate, if there are no source to highlight say None
	Step 9) For each Question generated, Highlight the name and version of the LLM that was used

	Return the results using this JSON schema:
		Assessment = {
		'Subject': string
		'Topic': string
		'Proficiency': string
		'Question': string
		'Answer': string
		'AllOptions':[]
		'Reasoning': string
		'Complexity': string
		'Source': string
		'LLMName': %s
		}
		Return: Array<Assessment>`

	return fmt.Sprintf(stepsPrompt, assessmentBankCount, complexity, proficiency, topic, subTopic, topic, llmName)

}

/* func testRun(ctx context.Context, promptString string, goRoute int) *genai.GenerateContentResponse {
	var llmName string

	if goRoute%2 == 0 {
		llmName = "gemini-1.5-flash"
	} else {
		llmName = "gemini-1.5-flash-8b"
	}

	// Access your API key as an environment variable (see "Set up your API key" above)
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}

	model := client.GenerativeModel(llmName)
	model.ResponseMIMEType = "application/json"
	const ChatTemperature float32 = 0.0
	temperature := ChatTemperature
	model.Temperature = &temperature

	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(systemPromptForValidation)},
	}

	resp, _ := model.GenerateContent(ctx, genai.Text(promptString))

	client.Close()

	return resp

} */

func workerforValidation(ctx context.Context, trackerforValdation chan empty, chanInputs chan []string, geminiResponseforValidation chan *genai.GenerateContentResponse, goRoute int) {

	var llmName string
	for chanInput := range chanInputs {

		llmName = "gemini-1.5-flash-8b"

		/* 		if goRoute%2 == 0 {
		   			llmName = "gemini-1.5-flash"
		   		} else {
		   			llmName = "gemini-1.5-flash-8b"
		   		} */

		// Access your API key as an environment variable (see "Set up your API key" above)
		client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			log.Fatal(err)
		}

		model := client.GenerativeModel(llmName)
		model.ResponseMIMEType = "application/json"
		const ChatTemperature float32 = 0.0
		temperature := ChatTemperature
		model.Temperature = &temperature

		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(systemPromptForValidation)},
		}

		resp, err := model.GenerateContent(ctx, genai.Text(chanInput[0]))
		if err != nil {
			geminiResponseforValidation <- nil
		} else {
			geminiResponseforValidation <- resp
		}

		client.Close()
	}
	var e empty
	trackerforValdation <- e

}

func worker(ctx context.Context, tracker chan empty, assessmentBankCount int, chanInputs chan []string, geminiResponse chan *genai.GenerateContentResponse, goRoute int) {

	var llmName string
	for chanInput := range chanInputs {

		llmName = "gemini-1.5-flash"

		promptString := getPromptRefined(assessmentBankCount, chanInput[0], chanInput[1], chanInput[2], chanInput[3], llmName)

		// Access your API key as an environment variable (see "Set up your API key" above)
		client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			log.Fatal(err)
		}

		model := client.GenerativeModel(llmName)
		model.ResponseMIMEType = "application/json"
		const ChatTemperature float32 = 0.0
		temperature := ChatTemperature
		model.Temperature = &temperature

		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(systemPrompt)},
		}

		/* 		model.ResponseSchema = &genai.Schema{
		   			Type:  genai.TypeArray,
		   			Items: &genai.Schema{Type: genai.TypeString},
		   		}
		*/
		/* 		model.SafetySettings = []*genai.SafetySetting{
			{
				Category:  genai.HarmCategoryDangerous,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategoryDangerousContent,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategoryHarassment,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategoryHateSpeech,
				Threshold: genai.HarmBlockNone,
			},
		} */

		resp, err := model.GenerateContent(ctx, genai.Text(promptString))
		if err != nil {
			geminiResponse <- nil
		} else {
			geminiResponse <- resp
		}

		client.Close()
	}
	var e empty
	tracker <- e

}

/* func printResponse(debug bool, resp *genai.GenerateContentResponse) {

	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if txt, ok := part.(genai.Text); ok {
					if debug {
						fmt.Println("----------------------------------------------------")
						fmt.Println(txt)
						fmt.Println("----------------------------------------------------")
					}
					var dataString []assessmentValidatedData
					if err := json.Unmarshal([]byte(txt), &dataString); err != nil {
						//log.Fatal(err)
						fmt.Println(err)
						continue
					}

					if debug {
						fmt.Println("----------------------------------------------------")
						for idx := 0; idx < len(dataString); idx++ {
							fmt.Println(dataString[idx].Question, dataString[idx].ValidatedAnswer, dataString[idx].ValidatedReasoning)
						}
						fmt.Println("----------------------------------------------------")
					}
				}
			}
		}
	}
} */

func getAllResponseMap(debug bool, resp *genai.GenerateContentResponse) (map[string]assessmentDataforMap, string) {

	resultsMap := make(map[string]assessmentDataforMap)

	var promptforValidation string

	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if txt, ok := part.(genai.Text); ok {
					var dataString []assessmentDataforMap
					if err := json.Unmarshal([]byte(txt), &dataString); err != nil {
						//log.Fatal(err)
						fmt.Println(err)
						continue
					} else {
						// fmt.Println("Len of dataString :", len(dataString))
						if dataString != nil {

							if debug {
								fmt.Println("-----------------------------------------------------------------")
								fmt.Println(dataString[0].Proficiency, dataString[0].Complexity, dataString[0].Topic, len(dataString))
								fmt.Println("-----------------------------------------------------------------")
							}

							promptforValidation = getPromptRefinedforValidation(dataString)

							for idx := 0; idx < len(dataString); idx++ {
								resultsMap[dataString[idx].Question] = dataString[idx]
							}
						}
					}

				}
			}
		}
	}

	return resultsMap, promptforValidation
}

func getAllValidatedResponseMap(resp *genai.GenerateContentResponse) map[string]assessmentValidatedData {

	validatedResultsMap := make(map[string]assessmentValidatedData)

	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if txt, ok := part.(genai.Text); ok {
					var dataString []assessmentValidatedData
					if err := json.Unmarshal([]byte(txt), &dataString); err != nil {
						fmt.Println(err)
						continue
					} else {
						// fmt.Println("Len of dataString :", len(dataString))
						if dataString != nil {
							for idx := 0; idx < len(dataString); idx++ {
								validatedResultsMap[dataString[idx].Question] = dataString[idx]
							}
						}
					}

				}
			}
		}
	}

	return validatedResultsMap
}

/* func getAllResponse(resp *genai.GenerateContentResponse) [][]assessmentData {

	var assessmentDataCollection [][]assessmentData

	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				// fmt.Println(part.(genai.Text))
				if txt, ok := part.(genai.Text); ok {
					var dataString []assessmentData
					if err := json.Unmarshal([]byte(txt), &dataString); err != nil {
						//log.Fatal(err)
						fmt.Println(err)
						continue
					}
					assessmentDataCollection = append(assessmentDataCollection, dataString)
				}
			}
		}
	}

	return assessmentDataCollection
} */

/* func tabWriteFile(assessmentDataCollectionFinal [][]assessmentData) {
	file, err := os.Create("generatedAssessments.csv")

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	writer := tabwriter.NewWriter(file, 0, 0, 1, ' ', 0)

	sep := "\t"

	fmt.Fprintln(writer, "Subject"+sep+"Topic"+sep+"Proficiency"+sep+"Complexity"+sep+"Question"+sep+"Option1"+sep+"Option2"+sep+"Option3"+sep+"Option4"+sep+"Answer"+sep+"Reasoning"+sep+"Source")

	for outIdx := range assessmentDataCollectionFinal {
		for inIdx := range assessmentDataCollectionFinal[outIdx] {

			fmt.Fprintln(writer,
				assessmentDataCollectionFinal[outIdx][inIdx].Subject+sep+assessmentDataCollectionFinal[outIdx][inIdx].Topic+sep+
					assessmentDataCollectionFinal[outIdx][inIdx].Proficiency+sep+assessmentDataCollectionFinal[outIdx][inIdx].Complexity+sep+
					assessmentDataCollectionFinal[outIdx][inIdx].Question+sep+assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[0]+sep+
					assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[1]+sep+assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[2]+sep+
					assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[3]+sep+assessmentDataCollectionFinal[outIdx][inIdx].Answer+sep+
					assessmentDataCollectionFinal[outIdx][inIdx].Reasoning+sep+assessmentDataCollectionFinal[outIdx][inIdx].Source)
		}
	}

	writer.Flush()
} */

func csvWriteFile(resultsMap map[string]assessmentDataforMap, fileName string) {

	file, err := os.Create(fileName)

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = ';'

	defer writer.Flush()

	var allDataStringSlice [][]string
	sep := ";"

	dataStringSlice := []string{"Subject" + sep + "Topic" + sep + "Proficiency" + sep + "Complexity" + sep +
		"Question" + sep + "Option1" + sep + "Option2" + sep + "Option3" + sep + "Option4" + sep + "Answer" + sep +
		"Reasoning" + sep + "Source" + sep + "LLMName" + sep + "ValidatedAnswer" + sep + "ValidatedReasoning" + sep + "ValidatedSelectedLLM"}

	allDataStringSlice = append(allDataStringSlice, dataStringSlice)

	for _, v := range resultsMap {

		dataStringSlice := []string{v.Subject + sep + v.Topic + sep +
			v.Proficiency + sep + v.Complexity + sep +
			v.Question + sep + v.AllOptions[0] + sep +
			v.AllOptions[1] + sep + v.AllOptions[2] + sep +
			v.AllOptions[3] + sep + v.Answer + sep +
			v.Reasoning + sep + v.Source + sep +
			v.LLMName + sep + v.ValidatedAnswer + sep + v.ValidatedReasoning + sep + v.ValidatedSelectedLLM}

		allDataStringSlice = append(allDataStringSlice, dataStringSlice)
	}

	writer.WriteAll(allDataStringSlice)

}

func updateMaps(debug bool, resultsMap map[string]assessmentDataforMap, allValidatedResultsMap map[string]assessmentValidatedData) (map[string]assessmentDataforMap, []assessmentDataforMap) {

	var localAssessmentDataforMap assessmentDataforMap
	var mismatchedDataString []assessmentDataforMap

	resultsMapCopy := make(map[string]assessmentDataforMap)

	correctAnswer := 0
	doesNotMatch := 0
	matchFound := false

	maps.Copy(resultsMapCopy, resultsMap)

	for kout, vout := range resultsMap {
		localAssessmentDataforMap = vout

		for kin, vin := range allValidatedResultsMap {
			if kin == kout {
				matchFound = true

				localAssessmentDataforMap.ValidatedAnswer = vin.ValidatedAnswer
				localAssessmentDataforMap.ValidatedReasoning = vin.ValidatedReasoning
				localAssessmentDataforMap.ValidatedSelectedLLM = "gemini-1.5-flash-8b"

				resultsMapCopy[kout] = localAssessmentDataforMap

				if localAssessmentDataforMap.ValidatedAnswer == localAssessmentDataforMap.Answer {
					correctAnswer++
				} else {
					mismatchedDataString = append(mismatchedDataString, localAssessmentDataforMap)
					if debug {
						fmt.Println("----------------------------------------------------")
						fmt.Println("Question")
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.Question)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.ValidatedAnswer, vout.Answer)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.ValidatedReasoning)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.Reasoning)
						fmt.Println("----------------------------------------------------")
					}
				}

				continue
			}
		}

		if !matchFound {
			mismatchedDataString = append(mismatchedDataString, localAssessmentDataforMap)

			doesNotMatch++
		}

		matchFound = false
	}

	if debug {
		fmt.Println("----------------------------------------------------")
		fmt.Println("Correct Anwers :", correctAnswer, " Total :", len(resultsMap), "  No Match : ", doesNotMatch, " Mismatched :", len(mismatchedDataString))
		fmt.Println("----------------------------------------------------")
	}

	return resultsMapCopy, mismatchedDataString
}

func validateAsessments(ctx context.Context, debug bool, promptforValidationList []string) map[string]assessmentValidatedData {
	const goRoutineCount int = 4
	var dataInput []string
	trackerforValdation := make(chan empty)
	chanInputsforValidation := make(chan []string)
	geminiResponseforValidation := make(chan *genai.GenerateContentResponse)
	var allValidatedResultsMap map[string]assessmentValidatedData

	// Create the jobs
	for i := 0; i < goRoutineCount; i++ {
		go workerforValidation(ctx, trackerforValdation, chanInputsforValidation, geminiResponseforValidation, i)
	}

	//get the completions
	go func() {
		for r := range geminiResponseforValidation {
			rvMap := getAllValidatedResponseMap(r)

			if allValidatedResultsMap == nil {
				allValidatedResultsMap = maps.Clone(rvMap)
			} else {
				maps.Copy(allValidatedResultsMap, rvMap)
			}
		}
		var e empty
		trackerforValdation <- e
	}()

	for pidx := range promptforValidationList {
		dataInput = nil

		dataInput = append(dataInput, promptforValidationList[pidx])

		chanInputsforValidation <- dataInput

	}

	close(chanInputsforValidation)
	for i := 0; i < goRoutineCount; i++ {
		<-trackerforValdation
	}
	close(geminiResponseforValidation)

	return allValidatedResultsMap
}

func generateAssessments(ctx context.Context, debug bool, record [][]string) (map[string]assessmentDataforMap, []string) {

	var resultsMap map[string]assessmentDataforMap
	var promptforValidationList []string

	const assessmentBankCount int = 20
	const goRoutineCount int = 4

	var profList []string

	profList = nil
	profList = append(profList, "Learner")
	profList = append(profList, "Practitioner")
	profList = append(profList, "Specialist")

	var complexityList []string

	complexityList = nil
	complexityList = append(complexityList, "Easy")
	complexityList = append(complexityList, "Medium")
	complexityList = append(complexityList, "Difficult")

	geminiResponse := make(chan *genai.GenerateContentResponse)

	// begin Implementation 2

	tracker := make(chan empty)
	chanInputs := make(chan []string)
	var dataInput []string

	// Create the jobs
	for i := 0; i < goRoutineCount; i++ {
		go worker(ctx, tracker, assessmentBankCount, chanInputs, geminiResponse, i)
	}

	//get the completions
	go func() {
		for r := range geminiResponse {

			rMap, promptforValidation := getAllResponseMap(debug, r)

			promptforValidationList = append(promptforValidationList, promptforValidation)

			if resultsMap == nil {
				resultsMap = maps.Clone(rMap)
			} else {
				maps.Copy(resultsMap, rMap)
			}
		}
		var e empty
		tracker <- e
	}()

	for idx := range profList {

		for cidx := range complexityList {

			for recordIteration := range record {
				dataInput = nil
				dataInput = append(dataInput, profList[idx])
				dataInput = append(dataInput, complexityList[cidx])
				dataInput = append(dataInput, record[recordIteration][0])
				dataInput = append(dataInput, record[recordIteration][1])

				chanInputs <- dataInput
			}
		}

	}

	close(chanInputs)
	for i := 0; i < goRoutineCount; i++ {
		<-tracker
	}
	close(geminiResponse)

	if debug {
		fmt.Println("----------------------------------------------------")
		fmt.Println("promptforValidationList:", len(promptforValidationList))
		fmt.Println("----------------------------------------------------")
	}

	return resultsMap, promptforValidationList

}

func main() {

	var debug bool = false

	// var sleepSeconds int = 60

	ctx := context.Background()

	csvfile, err := os.Open("TopicsforAssessmentGeneration.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}
	defer csvfile.Close()

	// Parse the file
	r := csv.NewReader(csvfile)

	record, _ := r.ReadAll()

	/* 	var resultsMap map[string]assessmentDataforMap
	   	var promptforValidationList []string
	   	const assessmentBankCount int = 20
	   	const goRoutineCount int = 8
	   	var profList []string

	   	profList = nil
	   	profList = append(profList, "Learner")
	   	profList = append(profList, "Practitioner")
	   	profList = append(profList, "Specialist")

	   	var complexityList []string

	   	complexityList = nil
	   	complexityList = append(complexityList, "Easy")
	   	complexityList = append(complexityList, "Medium")
	   	complexityList = append(complexityList, "Difficult")

	   	geminiResponse := make(chan *genai.GenerateContentResponse)

	   	// begin Implementation 2

	   	tracker := make(chan empty)
	   	chanInputs := make(chan []string)
	   	var dataInput []string

	   	// Create the jobs
	   	for i := 0; i < goRoutineCount; i++ {
	   		go worker(ctx, tracker, assessmentBankCount, chanInputs, geminiResponse, i)
	   	}

	   	//get the completions
	   	go func() {
	   		for r := range geminiResponse {
	   			// printResponse(r)
	   			// results = append(results, r)

	   			rMap, promptforValidation := getAllResponseMap(debug, r)

	   			promptforValidationList = append(promptforValidationList, promptforValidation)

	   			if resultsMap == nil {
	   				resultsMap = maps.Clone(rMap)
	   			} else {
	   				maps.Copy(resultsMap, rMap)
	   			}
	   		}
	   		var e empty
	   		tracker <- e
	   	}()

	   	for idx := range profList {

	   		for cidx := range complexityList {

	   			for recordIteration := range record {
	   				dataInput = nil
	   				dataInput = append(dataInput, profList[idx])
	   				dataInput = append(dataInput, complexityList[cidx])
	   				dataInput = append(dataInput, record[recordIteration][0])
	   				dataInput = append(dataInput, record[recordIteration][1])

	   				chanInputs <- dataInput
	   			}
	   		}

	   	}

	   	close(chanInputs)
	   	for i := 0; i < goRoutineCount; i++ {
	   		<-tracker
	   	}
	   	close(geminiResponse)

	   	if debug {
	   		fmt.Println("----------------------------------------------------")
	   		fmt.Println("promptforValidationList:", len(promptforValidationList))
	   		fmt.Println("----------------------------------------------------")
	   	} */

	/* const goRoutineCount int = 4
	var dataInput []string
	trackerforValdation := make(chan empty)
	chanInputsforValidation := make(chan []string)
	geminiResponseforValidation := make(chan *genai.GenerateContentResponse)
	var allValidatedResultsMap map[string]assessmentValidatedData

	// Create the jobs
	for i := 0; i < goRoutineCount; i++ {
		go workerforValidation(ctx, trackerforValdation, chanInputsforValidation, geminiResponseforValidation, i)
	}

	//get the completions
	go func() {
		for r := range geminiResponseforValidation {
			rvMap := getAllValidatedResponseMap(r)

			if allValidatedResultsMap == nil {
				allValidatedResultsMap = maps.Clone(rvMap)
			} else {
				maps.Copy(allValidatedResultsMap, rvMap)
			}
		}
		var e empty
		trackerforValdation <- e
	}()

	for pidx := range promptforValidationList {
		dataInput = nil

		dataInput = append(dataInput, promptforValidationList[pidx])

		chanInputsforValidation <- dataInput
	}

	close(chanInputsforValidation)
	for i := 0; i < goRoutineCount; i++ {
		<-trackerforValdation
	}
	close(geminiResponseforValidation)

	if debug {
		fmt.Println("----------------------------------------------------")
		fmt.Println("Len of Validated List :", len(allValidatedResultsMap), "Len of Map  :", len(resultsMap))
		fmt.Println("----------------------------------------------------")
	} */

	/* var mismatchedDataString []assessmentDataforMap
	var localAssessmentDataforMap assessmentDataforMap
	correctAnswer := 0
	doesNotMatch := 0
	matchFound := false

	for kout, vout := range resultsMap {
		localAssessmentDataforMap = vout

		for kin, vin := range allValidatedResultsMap {
			if kin == kout {
				matchFound = true

				localAssessmentDataforMap.ValidatedAnswer = vin.ValidatedAnswer
				localAssessmentDataforMap.ValidatedReasoning = vin.ValidatedReasoning
				localAssessmentDataforMap.ValidatedSelectedLLM = "gemini-1.5-flash-8b"

				resultsMap[kout] = localAssessmentDataforMap

				if localAssessmentDataforMap.ValidatedAnswer == localAssessmentDataforMap.Answer {
					correctAnswer++
				} else {
					mismatchedDataString = append(mismatchedDataString, localAssessmentDataforMap)
					if debug {
						fmt.Println("----------------------------------------------------")
						fmt.Println("Question")
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.Question)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.ValidatedAnswer, vout.Answer)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.ValidatedReasoning)
						fmt.Println("----------------------------------------------------")
						fmt.Println(vout.Reasoning)
						fmt.Println("----------------------------------------------------")
					}
				}

				continue
			}
		}

		if !matchFound {
			mismatchedDataString = append(mismatchedDataString, localAssessmentDataforMap)

			doesNotMatch++
		}

		matchFound = false
	}

	if debug {
		fmt.Println("----------------------------------------------------")
		fmt.Println("Correct Anwers :", correctAnswer, " Total :", len(resultsMap), "  No Match : ", doesNotMatch, " Mismatched :", len(mismatchedDataString))
		fmt.Println("----------------------------------------------------")
	}
	*/

	fmt.Println("Generating Assessments Started")
	resultsMap, promptforValidationList := generateAssessments(ctx, debug, record)
	fmt.Println("Generating Assessments Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Flushing Assessments Started")
	csvWriteFile(resultsMap, "generatedAssessments.csv")
	fmt.Println("Flushing Assessments Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Validating Assessments Started")
	allValidatedResultsMap := validateAsessments(ctx, debug, promptforValidationList)
	fmt.Println("Validating Assessments Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Updating Maps Started")
	resultsMap, mismatchedDataString := updateMaps(debug, resultsMap, allValidatedResultsMap)
	fmt.Println("Updating Maps Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Round 1 Stats")
	fmt.Println("----------------------------------------------------")
	fmt.Println("Len of Validated List :", len(allValidatedResultsMap), "Len of Map  :", len(resultsMap), " Mismatched :", len(mismatchedDataString))
	fmt.Println("----------------------------------------------------")
	fmt.Println("Round 1 Stats")

	csvWriteFile(resultsMap, "generatedAssessmentsValidated-1.csv")

	fmt.Println("Validating Assessments Round 2 Started")
	promptforValidationList = nil
	promptforValidationList = append(promptforValidationList, getPromptRefinedforValidation(mismatchedDataString))
	fmt.Println("Validating Assessments Round 2 Started")

	fmt.Println("Validating Assessments Round 2 Started")
	allValidatedResultsMap = validateAsessments(ctx, debug, promptforValidationList)
	fmt.Println("Validating Assessments Round 2 Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Updating Maps Round 2 Started")
	resultsMap, mismatchedDataString = updateMaps(debug, resultsMap, allValidatedResultsMap)
	fmt.Println("Updating Maps Round 2 Done")

	time.Sleep(60 * time.Second)

	fmt.Println("Round 2 Stats")
	fmt.Println("----------------------------------------------------")
	fmt.Println("Len of Validated List :", len(allValidatedResultsMap), "Len of Map  :", len(resultsMap), " Mismatched :", len(mismatchedDataString))
	fmt.Println("----------------------------------------------------")
	fmt.Println("Round 2 Stats")

	fmt.Println("Flushing Assessments Started")
	csvWriteFile(resultsMap, "generatedAssessmentsValidated-2.csv")
	fmt.Println("Flushing Assessments Done")

}
