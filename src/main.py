from asyncio import run

from core.stage_1.main_topics import GeneralDiscussionAnalysis
from core.stage_2.treatment_vars import TreatmentVars
from core.stage_4.treatment_options import TreatmentOptions
from core.stage_4.sent_analysis import SentimentAnalysis

from src.core.preprocess_data import Preprocess


async def main():
    get_data = Preprocess()
    compute_lda = GeneralDiscussionAnalysis()
    treatment_opt = TreatmentVars()
    text_similarity_bert: TreatmentOptions = TreatmentOptions()
    sent_analysis: SentimentAnalysis = SentimentAnalysis()

    # data = await get_data.fully_preprocessed_data()
    # bert_check = await treatment_opt.combine_visualization()
    # # print(bert_check)

    # sent_analysis = await sent_analysis.possitive_sentiment()
    # print(sent_analysis)

    lsa_output = await treatment_opt.combine_visualization()
    print(lsa_output)
    #
    # bio_ner = await treatment_opt.bio_ner()
    # print(bio_ner)

if __name__ == '__main__':
    run(main())
