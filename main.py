import torch
import json
import torch.backends.cudnn as cudnn
from utils import logger, load_documents
# import wandb
from train_config import args


def main():
    if torch.cuda.device_count() > 0:
        cudnn.benchmark = True

    from train_config import TrainingArguments
    args.use_rdrop = False
    train_args = TrainingArguments
    train_args.learning_rate = args.learning_rate
    train_args.train_batch_size = args.train_batch_size
    train_args.eval_batch_size = args.eval_batch_size
    train_args.num_cand = args.num_candidates
    train_args.epochs = args.epochs
    train_args.eval_every_n_intervals=2
    train_args.log_every_n_intervals=30

    all_documents = {}      # doc_id/ entity_id to entity
    document_path = args.document_files[0].split(",")
    for input_file_path in document_path:
        all_documents.update(load_documents(input_file_path))


    # wandb.init(settings=wandb.Settings(start_method="fork"), project="BMKG", entity="marvinpeng", config=vars(args))
    # wandb.config.update(args, allow_val_change=True)
    if args.model_type == 'cross_encoder' and (args.lora or args.bitfit or args.adapter):
        from trainer.adapter_cross_encoder_trainer import Trainer
        from preprocessing.preprocess_data import EntityLinkingSet

        train_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.train_mentions_file,
            tfidf_candidates_file=args.train_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=True)

        eval_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.eval_mentions_file,
            tfidf_candidates_file=args.eval_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)

        trainer = Trainer(
        args,
        pretrained_model_path=args.pretrained_model_path,
        eval_model_path=args.eval_model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        learning_rate=args.learning_rate,
        train_args=train_args,
        num_workers=4,
        model_type='cross_encoder',
    )

    elif args.model_type == 'cross_encoder':
        from trainer.cross_encoder_trainer import Trainer
        from preprocessing.preprocess_data import EntityLinkingSet

        train_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.train_mentions_file,
            tfidf_candidates_file=args.train_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=True)

        eval_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.eval_mentions_file,
            tfidf_candidates_file=args.eval_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)

        trainer = Trainer(
        args,
        pretrained_model_path=args.pretrained_model_path,
        eval_model_path=args.eval_model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        learning_rate=args.learning_rate,
        train_args=train_args,
        num_workers=4,
        model_type='cross_encoder',
    )
    elif args.model_type == "contrastive" and (args.lora or args.bitfit or args.adapter):
        from trainer.adapter_bi_encoder_trainer import Trainer
        from preprocessing.bi_preprocess_data import EntityLinkingSet

        train_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.train_mentions_file,
            tfidf_candidates_file=args.train_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=True)

        eval_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.eval_mentions_file,
            tfidf_candidates_file=args.eval_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)

        trainer = Trainer(
        args,
        pretrained_model_path=args.pretrained_model_path,
        learning_rate=args.learning_rate,
        training_epochs=args.epochs,
        num_workers=args.num_workers,
        num_candidates=args.num_candidates,
        eval_model_path=args.eval_model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_args=train_args,
        model_type=args.model_type,
        use_tf_idf_negatives=args.use_tf_idf_negatives,
        use_in_batch_mention_negatives=args.use_mention_negatives,
        use_rdrop=args.use_rdrop,
        margin=args.margin
    )
    elif "contrastive" in args.model_type:
        from trainer.bi_encoder_trainer import Trainer
        from preprocessing.bi_preprocess_data import EntityLinkingSet

        train_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.train_mentions_file,
            tfidf_candidates_file=args.train_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=True)

        eval_dataset = EntityLinkingSet(
            # document_files=args.document_files,
            document_files=all_documents,
            mentions_path=args.eval_mentions_file,
            tfidf_candidates_file=args.eval_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)

        trainer = Trainer(
        args,
        pretrained_model_path=args.pretrained_model_path,
        learning_rate=args.learning_rate,
        training_epochs=args.epochs,
        num_workers=args.num_workers,
        num_candidates=args.num_candidates,
        eval_model_path=args.eval_model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_args=train_args,
        model_type=args.model_type,
        use_tf_idf_negatives=args.use_tf_idf_negatives,
        use_in_batch_mention_negatives=args.use_mention_negatives,
        use_rdrop=args.use_rdrop,
        margin=args.margin
    )
    else:
        raise NotImplementedError

    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.run()

if __name__ == '__main__':
    main()
