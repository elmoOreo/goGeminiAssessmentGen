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

func getPrompt(assessmentBank int, prof string, topic string, subTopic string) string {

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

	return fmt.Sprintf(promptTemplate, assessmentBank, prof, subTopic, topic)

}

func getPromptRefined(assessmentBank int, prof string, topic string, subTopic string) string {

	promptTemplate := `
	You are an AI Guru and an expert in AI literature.  You are tasked to generate a set of multiple choice assessments for evaluating a talent.

	The talent can belong to one of the following proficiencies in the increasing order either Learner, Practitioner or Specialist 
	1) Learner      : A Learner is at level 0. A Learner is aware of the concept and has little or minimal practical experience 
	2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable on the concept anad has meaningful practical experience of implementing the concept.
	3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has great practical experience of implementing the concept.

	Please do not hallucinate, if you are not aware, please say it so in courteous fashion. Please do not share anything that can be construed as harmful. You will be formulating a question and set of possible answers that can be presented for evaluation.

	You will build an assessment bank of %d questions for evaluating the Proficiency of a %s on the Subject of %s"

	The following will be the steps for each question

	Step 1) Ask a relevant question on the Topic of %s. 
	Step 2) If the question is a repeat, ask a different relevant question
	Step 3) For each question generated, List the answer having a maximum length of no more than 5 words to the question.
	Step 4) For each question generated, Generate 3 other similar answers having a maximum length of no more than 5 words. 
	Step 5) For each question generated, Return the answers generated in Step 3 and Step 4 as a single list.
	Step 6) For each question generated, Articulate in detail why the answer in Step 3 is the right answer for the question
	Step 7) For each question generated, Estimate the complexity of the question in terms of Easy, Medium or Difficult
	Step 8) For each question generated, Highlight the source if any from which the question was articulated, do not hallucinate, if there are no source to highlight say none

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

	return fmt.Sprint(promptTemplate, assessmentBank, prof, topic, subTopic)

}

func worker(ctx context.Context, assessmentBank int, prof string, topic string, subTopic string, geminiResponse chan *genai.GenerateContentResponse, wg *sync.WaitGroup) {

	defer wg.Done()

	promptString := getPromptRefined(assessmentBank, prof, topic, subTopic)

	// Access your API key as an environment variable (see "Set up your API key" above)
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-1.5-flash")
	model.ResponseMIMEType = "application/json"
	const ChatTemperature float32 = 0.0
	temperature := ChatTemperature
	model.Temperature = &temperature

	resp, err := model.GenerateContent(ctx, genai.Text(promptString))
	if err != nil {
		log.Fatal(err)
	}

	geminiResponse <- resp

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
func main() {

	var results []*genai.GenerateContentResponse

	ctx := context.Background()

	var assessmentDataCollectionFinal [][]assessmentData

	const questionCount int = 10

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

	var wg sync.WaitGroup
	geminiResponse := make(chan *genai.GenerateContentResponse)

	for idx := range profList {
		for recordIteration := range record {
			wg.Add(1)
			go worker(ctx, questionCount, profList[idx], record[recordIteration][0], record[recordIteration][1], geminiResponse, &wg)
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
