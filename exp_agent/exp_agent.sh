#!/bin/bash
source config.sh

# mainブランチ同期
git checkout main
git pull origin main
# agent用の親ブランチ作成・チェックアウト
parent_branch=agent$(date +"%Y%m%d%H%M%S")
git checkout -b $parent_branch
git push -u origin $parent_branch

# 実験ループ
while true; do
    # githubのissuesから実験内容を取得
    IFS=$'\n' read -d '' -r -a values < <(python get_action_item.py --owner $GITHUB_OWNNER --repo $GITHUB_REPO && printf '\0')
    issue_number=${values[0]}
    action_item=${values[1]}

    # openのissuesがなかったらループを抜けて終了
    if [ "$issue_number" -eq -1 ]; then
        break
    fi

    # 子ブランチ作成・チェックアウト
    child_branch=$parent_branch-$issue_number
    git checkout -b $child_branch

    # 実験前のbest cvスコアを取得
    prev_best_score=$(python get_current_best_score.py --score_name $WANDB_SCORE_NAME --direction $SCORE_DIRECTION)

    # 実験内容から新しいexpコードを生成
    python generate_new_exp_code.py --exp_code_path $EXP_CODE_PATH --action_item $action_item --llm_model $LLM_MODEL

    # expコードの実行
    python $EXP_CODE_PATH

    # wandbのNotesに実験内容を書き込み
    python write_wandb_latest_run_notes.py --notes $action_item

    # 実験後のbest cvスコアを取得
    new_best_score=$(python get_current_best_score.py --score_name $WANDB_SCORE_NAME --direction $SCORE_DIRECTION)

    # 変更点をgithubに登録
    git add -u
    git commit -m "$action_item #$issue_number"
    git push -u origin $child_branch
    python close_github_issue.py --owner $GITHUB_OWNNER --repo $GITHUB_REPO --issue_number $issue_number

    git checkout $parent_branch

    # cvを更新したら子ブランチの内容を親ブランチに反映
    result=$(python check_score_improved.py --prev $prev_best_score --new $new_best_score --direction $SCORE_DIRECTION)
    if [ "$result" -eq 1 ]; then
        git merge $child_branch
        git push origin $parent_branch
    fi
done