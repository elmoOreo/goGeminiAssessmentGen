package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"text/tabwriter"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type empty struct{}

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
}

func getPrompt(assessmentBankCount int, prof string, topic string, subTopic string) string {

	promptTemplate := `
	You are an AI Guru and an expert in AI literature.  You are tasked to generate a set of multiple choice assessments for evaluating a talent.

	The talent can belong to one of the proficiencies in the ascending order
	1) Learner       : A Learner is at level 1. A Learner is aware of the concept and has little or minimal practical experience 
	2) Practitioner  : A Practitioner is at level 2 and has more knowledge than a Learner. A Practitioner is knowledgable on the concept anad has meaningful practical experience of implementing the concept.
	3) Specialist    : A Specialist is at level 3 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has great practical experience of implementing the concept.

	Please do not hallucinate, if you are not aware, please say it so in courteous fashion. Please do not share anything that can be construed as harmful. You will be formulating a question and set of possible answers that can be presented for evaluation.
	The following will be your task in steps
	Step 1) Ask %d relevant questions for evaluating a talent who is a %s on the Subject of %s in the Topic of %s. Ensure all questions are unique and not repeated. If a question has already been asked, revisit the question
	Step 2) For each question generated in Step 1, List the answer having a maximum length of no more than 5 words to the question in Step 1.
	Step 3) For each question generated in Step 1, Generate 3 other similar answers having a maximum length of no more than 5 words as in Step 2. 
	Step 4) For each question generated in Step 1, Return the answers generated in Step 2 and Step 3 as a single list.
	Step 5) For each question generated in Step 1, Articulate in detail why the answer in Step 2 is the right answer for the question in Step 1
	Step 6) For each question generated in Step 1, Estimate the complexity of the question in terms of easy, medium, difficult
	Step 7) For each question generated in Step 1, Highlight the source if any from which the question was articulated, do not hallucinate, if there are no source to highlight say none

	Return the results using this JSON schema: 
		Assessment = {
		"Subject": str
		"Topic": str
		"Proficiency": str
		"Question": str
		"Answer": str
		"AllOptions":[]
		"Reasoning": str
		"Complexity": str
		"Source": str
		}
	 Return: Array<Assessment>
	`

	return fmt.Sprintf(promptTemplate, assessmentBankCount, prof, subTopic, topic)

}

func getPromptRefined(assessmentBankCount int, prof string, topic string, subTopic string) string {

	promptTemplate := `
	You are an AI Guru and an expert in AI literature.  You are tasked to generate a set of multiple choice assessments for evaluating a talent.

	The talent can belong to one of the following proficiencies in the increasing order either Learner, Practitioner or Specialist 
	1) Learner      : A Learner is at level 0. A Learner is aware of the concept and has little or minimal practical experience 
	2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable on the concept anad has meaningful practical experience of implementing the concept over a few years.
	3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has meaningful practical experience of implementing the concept over many years.

	Please do not hallucinate, if you are not aware, please say it so in courteous fashion. Please do not share anything that can be construed as harmful. You will be formulating a question and set of possible answers that can be presented for evaluation.

	You will build an assessment bank of %d questions for evaluating the Proficiency of a %s on the Subject of %s"

	The following will be the steps for each question

	Step 1) Ask a relevant question on the Topic of %s. 
	Step 2) If the question is a repeat, ask a different relevant question.
	Step 3) For the question generated, List the answer having a maximum length of no more than 5 words.
	Step 4) For the question generated, Generate 3 other similar answers having a maximum length of no more than 5 words. 
	Step 5) For the question generated, Return the answers generated in Step 3 and Step 4 as a single list.
	Step 6) For the question generated, Articulate in detail why the answer in Step 3 is the right answer.
	Step 7) For the question generated, Estimate the complexity of the question in terms of Easy, Medium or Difficult.
	Step 8) For the question generated, Highlight the source if any from which the question was articulated, do not hallucinate, if there are no source to highlight say none.

	Return the results using this JSON schema: 
		Assessment = {
		"Subject": %s
		"Topic": %s
		"Proficiency": %s
		"Question": str
		"Answer": str
		"AllOptions":[]
		"Reasoning": str
		"Complexity": str
		"Source": str
		}
	 Return: Array<Assessment>
	`

	return fmt.Sprint(promptTemplate, assessmentBankCount, prof, topic, subTopic, topic, subTopic, prof)

}

func worker(ctx context.Context, assessmentBankCount int, prof string, topic string, subTopic string, geminiResponse chan *genai.GenerateContentResponse, wg *sync.WaitGroup) {

	defer wg.Done()

	promptString := getPromptRefined(assessmentBankCount, prof, topic, subTopic)

	// Access your API key as an environment variable (see "Set up your API key" above)
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-1.5-flash")
	model.ResponseMIMEType = "application/json"
	const ChatTemperature float32 = 0.5
	temperature := ChatTemperature
	model.Temperature = &temperature

	resp, err := model.GenerateContent(ctx, genai.Text(promptString))
	if err != nil {
		log.Fatal(err)
	}

	geminiResponse <- resp

}

func worker2(ctx context.Context, tracker chan empty, assessmentBankCount int, chanInputs chan []string, geminiResponse chan *genai.GenerateContentResponse) {

	for chanInput := range chanInputs {
		promptString := getPromptRefined(assessmentBankCount, chanInput[0], chanInput[1], chanInput[2])

		// Access your API key as an environment variable (see "Set up your API key" above)
		client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			log.Fatal(err)
		}

		model := client.GenerativeModel("gemini-1.5-flash")
		model.ResponseMIMEType = "application/json"

		resp, err := model.GenerateContent(ctx, genai.Text(promptString))
		if err != nil {
			log.Fatal(err)
		}

		geminiResponse <- resp

		client.Close()
	}
	var e empty
	tracker <- e

}

func getAllResponse(resp *genai.GenerateContentResponse) [][]assessmentData {

	var assessmentDataCollection [][]assessmentData

	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				// fmt.Println(part.(genai.Text))
				if txt, ok := part.(genai.Text); ok {
					var dataString []assessmentData
					if err := json.Unmarshal([]byte(txt), &dataString); err != nil {
						log.Fatal(err)
					}
					assessmentDataCollection = append(assessmentDataCollection, dataString)
				}
			}
		}
	}

	return assessmentDataCollection
}

func tabWriteFile(assessmentDataCollectionFinal [][]assessmentData) {
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
}

func csvWriteFile(assessmentDataCollectionFinal [][]assessmentData) {

	file, err := os.Create("generatedAssessments.csv")

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = ';'

	defer writer.Flush()

	var allDataStringSlice [][]string
	sep := ";"

	dataStringSlice := []string{"Subject" + sep + "Topic" + sep + "Proficiency" + sep + "Complexity" + sep + "Question" + sep + "Option1" + sep + "Option2" + sep + "Option3" + sep + "Option4" + sep + "Answer" + sep + "Reasoning" + sep + "Source"}

	allDataStringSlice = append(allDataStringSlice, dataStringSlice)

	for outIdx := range assessmentDataCollectionFinal {
		for inIdx := range assessmentDataCollectionFinal[outIdx] {
			dataStringSlice := []string{assessmentDataCollectionFinal[outIdx][inIdx].Subject + sep + assessmentDataCollectionFinal[outIdx][inIdx].Topic + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Proficiency + sep + assessmentDataCollectionFinal[outIdx][inIdx].Complexity + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Question + sep + assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[0] + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[1] + sep + assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[2] + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[3] + sep + assessmentDataCollectionFinal[outIdx][inIdx].Answer + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Reasoning + sep + assessmentDataCollectionFinal[outIdx][inIdx].Source}

			allDataStringSlice = append(allDataStringSlice, dataStringSlice)
		}
	}

	writer.WriteAll(allDataStringSlice)

}

func main() {

	var results []*genai.GenerateContentResponse

	ctx := context.Background()

	var assessmentDataCollectionFinal [][]assessmentData

	const assessmentBankCount int = 20
	const goRoutineCount int = 5

	csvfile, err := os.Open("TopicsforAssessmentGeneration.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}
	defer csvfile.Close()

	// Parse the file
	r := csv.NewReader(csvfile)

	record, _ := r.ReadAll()

	var profList []string

	profList = append(profList, "Learner")
	profList = append(profList, "Practitioner")
	profList = append(profList, "Specialist")

	geminiResponse := make(chan *genai.GenerateContentResponse)

	// begin Implementation 2

	tracker := make(chan empty)
	chanInputs := make(chan []string)
	var dataInput []string

	// Create the jobs
	for i := 0; i < goRoutineCount; i++ {
		go worker2(ctx, tracker, assessmentBankCount, chanInputs, geminiResponse)
	}

	//get the completions
	go func() {
		for r := range geminiResponse {
			// printResponse(r)
			results = append(results, r)
		}
		var e empty
		tracker <- e
	}()

	for idx := range profList {
		for recordIteration := range record {
			dataInput = nil
			dataInput = append(dataInput, profList[idx])
			dataInput = append(dataInput, record[recordIteration][0])
			dataInput = append(dataInput, record[recordIteration][1])

			chanInputs <- dataInput

		}
	}

	close(chanInputs)
	for i := 0; i < goRoutineCount; i++ {
		<-tracker
	}
	close(geminiResponse)

	assessmentDataCollectionFinal = nil
	for r := range results {
		assessmentDataCollectionFinal = append(assessmentDataCollectionFinal, getAllResponse(results[r])...)
	}

	csvWriteFile(assessmentDataCollectionFinal)

	// end Implementation 2

	// begin Implementation 1

	/* var wg sync.WaitGroup

	for idx := range profList {
		for recordIteration := range record {
			wg.Add(1)
			go worker(ctx, assessmentBankCount, profList[idx], record[recordIteration][0], record[recordIteration][1], geminiResponse, &wg)
		}
	}

	go func() {
		for r := range geminiResponse {
			// printResponse(r)
			results = append(results, r)
		}
	}()

	wg.Wait()
	close(geminiResponse)

	for r := range results {
		assessmentDataCollectionFinal = append(assessmentDataCollectionFinal, getAllResponse(results[r])...)
	}

	csvWriteFile(assessmentDataCollectionFinal) */

	// end Implementation 1

}
