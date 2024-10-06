package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"

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
	LLMName     string
}

/* func getPrompt(assessmentBankCount int, prof string, topic string, subTopic string) string {

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
		"Subject": string
		"Topic": string
		"Proficiency": string
		"Question": string
		"Answer": string
		"AllOptions":[]
		"Reasoning": string
		"Complexity": string
		"Source": string
		}
	 Return: Array<Assessment>
	`

	return fmt.Sprintf(promptTemplate, assessmentBankCount, prof, subTopic, topic)

} */

func getPromptRefined(assessmentBankCount int, proficiency string, complexity string, topic string, subTopic string, llmName string) string {

	var promptTemplate string

	/* promptTemplate := `You are an AI Guru and an expert in AI literature.

	You are tasked to generate a set of multiple choice assessments for evaluating a Talent based on their proficiency on multiple Topics in a Subject area.
	The assessments against each Proficiency will have a mix of Easy, Medium or Difficult questions designed to test the Talent. The typical distribution would be 20 percent Easy,30 percent Medium and 40 percent Difficult.

	The Talent can belong to one of the following Proficiencies in the increasing order either Learner, Practitioner or Specialist
	1) Learner      : A Learner is at level 0. A Learner is aware of the basic concepts and has little or minimal practical experience.
	2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable and has an advanced understanding of the concept and has meaningful practical experience of implementing the concept over a few years.
	3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has meaningful practical experience of implementing the concept over many years.

	Please do not hallucinate, if you are not aware, please say it so in courteous fashion.
	Please do not share anything that can be construed as harmful.

	You will build an assessment bank of %d questions for evaluating the Proficiency of a %s on different topics under the Subject of %s

	The following will be the steps for each question

	Step 1) Ask a relevant Question on the Topic of %s within the context of %s.
	Step 2) If the Question is a repeat, ask a different relevant Question
	Step 3) For each Question generated, List the Answer having a maximum length of no more than 5 words to the Question.
	Step 4) For each Question generated, Generate 3 other similar answers having a maximum length of no more than 5 words.
	Step 5) For each Question generated, Return the answers generated in Step 3 and Step 4 as a single list as AllOptions.
	Step 6) For each Question generated, Articulate in detail why the Answer in Step 3 is the right Answer for the Question as Reasoning.
	Step 7) For each Question generated, Estimate the Complexity of the question in terms of Easy, Medium or Difficult.
	Step 8) For each Question generated, Highlight the Source if any from which the question was articulated, do not hallucinate, if there are no source to highlight say None

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
		}
	 Return: Array<Assessment>
	` */

	// 	The typical distribution would be 20 percent Easy,30 percent Medium and 40 percent Difficult.

	systemPrompt := `You are an AI Guru and an expert in AI literature. You are tasked to generate a set of multiple choice assessments for evaluating a Talent based on their proficiency on multiple Topics in a particular Subject area.
	The talent can belong to one of the following Proficiencies in the increasing order of expertise either a Learner, Practitioner or Specialist. The following is the definition of each Proficiency
	1) Learner      : A Learner is at a level 0 or foundation level. A Learner is aware of the basic concepts and has little or minimal practical experience.
	2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable and has an advanced understanding of the concept and has meaningful practical experience of implementing the concept over a few years.
	3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has meaningful practical experience of implementing the concept over many years.

	The assessments against each proficiency on a topic will have a mix of Easy, Medium or Difficult questions designed to test the Talent. The following is the definition of each level of Complexity
	1) Easy     	: Easy questions fall around basic knowledge or understanding. There is no ambiguity or hidden meanings. It often involves a single calculation or task.
	2) Medium 		: Medium questions demand a deeper understanding of the subject matter. It involes reasoning.
	3) Difficult	: Difficult questions centre around combination different ideas or principles. It demands analysis, evaluation, and synthesis of information.

	Please do not hallucinate, if you are not aware, please say it so in courteous fashion. Please do not share anything that can be construed as harmful.`

	/* 	learnerPrompt := "A Learner is aware of the basic concepts and has little or minimal practical experience."
	   	practitionerPrompt := "A Practitioner is knowledgable, has an understanding of the advanced concepts and has meaningful practical experience of implementing the concept over a few years."
	   	specialistPrompt := "A Specialist is an Expert on the concept, has a deep understanding of the advanced concepts and has meaningful practical experience of implementing the concept over many years."
	*/
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

	promptTemplate = fmt.Sprintf("%s \n  %s", systemPrompt, stepsPrompt)

	/* 	if proficiency == "Learner" {

	   		promptTemplate = fmt.Sprintf("%s \n %s \n %s", systemPrompt, learnerPrompt, stepsPrompt)

	   	} else if proficiency == "Practitioner" {

	   		promptTemplate = fmt.Sprintf("%s \n %s \n %s", systemPrompt, practitionerPrompt, stepsPrompt)

	   	} else if proficiency == "Specialist" {

	   		promptTemplate = fmt.Sprintf("%s \n %s \n %s", systemPrompt, specialistPrompt, stepsPrompt)

	   	} */

	return fmt.Sprintf(promptTemplate, assessmentBankCount, complexity, proficiency, topic, subTopic, topic, llmName)

}

/* func worker(ctx context.Context, assessmentBankCount int, prof string, topic string, subTopic string, geminiResponse chan *genai.GenerateContentResponse, wg *sync.WaitGroup) {

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
	const ChatTemperature float32 = 0.0
	temperature := ChatTemperature
	model.Temperature = &temperature

	model.ResponseSchema = &genai.Schema{
		Type:  genai.TypeArray,
		Items: &genai.Schema{Type: genai.TypeString},
	}

	resp, err := model.GenerateContent(ctx, genai.Text(promptString))
	if err != nil {
		log.Fatal(err)
	}

	geminiResponse <- resp

} */

func worker2(ctx context.Context, tracker chan empty, assessmentBankCount int, chanInputs chan []string, geminiResponse chan *genai.GenerateContentResponse, goRoute int) {

	var llmName string
	for chanInput := range chanInputs {

		if goRoute%2 == 0 {
			llmName = "gemini-1.5-flash"
		} else {
			llmName = "gemini-1.5-flash-8b"
		}

		promptString := getPromptRefined(assessmentBankCount, chanInput[0], chanInput[1], chanInput[2], chanInput[3], llmName)

		/* fmt.Println("------Prompt------")
		fmt.Println(promptString)
		fmt.Println("------Prompt------") */

		// Access your API key as an environment variable (see "Set up your API key" above)
		client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
		if err != nil {
			log.Fatal(err)
		}

		model := client.GenerativeModel(llmName)
		model.ResponseMIMEType = "application/json"
		const ChatTemperature float32 = 1.0
		temperature := ChatTemperature
		model.Temperature = &temperature

		/* 		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(`
				You are an AI Guru and an expert in AI literature.

				You are tasked to generate a set of multiple choice assessments for evaluating a Talent based on their proficiency on multiple Topics in a Subject area.
				The assessments against each Proficiency will have a mix of Easy, Medium or Difficult questions designed to test the Talent. The typical distribution would be 20 percent Easy,30 percent Medium and 40 percent Difficult.

				The Talent can belong to one of the following Proficiencies in the increasing order either Learner, Practitioner or Specialist
				1) Learner      : A Learner is at level 0. A Learner is aware of the basic concepts and has little or minimal practical experience.
				2) Practitioner : A Practitioner is at level 1 and has more knowledge than a Learner. A Practitioner is knowledgable and has an advanced understanding of the concept and has meaningful practical experience of implementing the concept over a few years.
				3) Specialist   : A Specialist is at level 2 and has more knowledge than a Practitioner. A Specialist is a GURU on the concept and has meaningful practical experience of implementing the concept over many years.

				Please do not hallucinate, if you are not aware, please say it so in courteous fashion.
				Please do not share anything that can be construed as harmful. `)},
		} */

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

func getAllResponse(resp *genai.GenerateContentResponse) [][]assessmentData {

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
					fmt.Println(dataString[0].Proficiency, dataString[0].Complexity, dataString[0].Topic, len(dataString))
					assessmentDataCollection = append(assessmentDataCollection, dataString)
				}
			}
		}
	}

	return assessmentDataCollection
}

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

	dataStringSlice := []string{"Subject" + sep + "Topic" + sep + "Proficiency" + sep + "Complexity" + sep + "Question" + sep + "Option1" + sep + "Option2" + sep + "Option3" + sep + "Option4" + sep + "Answer" + sep + "Reasoning" + sep + "Source" + sep + "LLMName"}

	allDataStringSlice = append(allDataStringSlice, dataStringSlice)

	for outIdx := range assessmentDataCollectionFinal {
		for inIdx := range assessmentDataCollectionFinal[outIdx] {
			dataStringSlice := []string{assessmentDataCollectionFinal[outIdx][inIdx].Subject + sep + assessmentDataCollectionFinal[outIdx][inIdx].Topic + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Proficiency + sep + assessmentDataCollectionFinal[outIdx][inIdx].Complexity + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Question + sep + assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[0] + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[1] + sep + assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[2] + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].AllOptions[3] + sep + assessmentDataCollectionFinal[outIdx][inIdx].Answer + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].Reasoning + sep + assessmentDataCollectionFinal[outIdx][inIdx].Source + sep +
				assessmentDataCollectionFinal[outIdx][inIdx].LLMName}

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
	const goRoutineCount int = 4

	csvfile, err := os.Open("TopicsforAssessmentGeneration.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}
	defer csvfile.Close()

	// Parse the file
	r := csv.NewReader(csvfile)

	record, _ := r.ReadAll()

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
		go worker2(ctx, tracker, assessmentBankCount, chanInputs, geminiResponse, i)
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
