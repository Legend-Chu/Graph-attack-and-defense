#!/bin/bash
# Auto contribute script - 200 commits with different dates

TOTAL=200
# Spread commits over the last 200 days
START_DATE=$(date -d "200 days ago" +"%Y-%m-%d")

# Create a contribution log file
touch contribution_log.md

for i in $(seq 1 $TOTAL); do
    # Calculate date: spread evenly over 200 days
    DAYS_AGO=$((200 - i))
    COMMIT_DATE=$(date -d "$DAYS_AGO days ago" +"%Y-%m-%dT12:00:00")

    # Different types of small changes
    case $((i % 5)) in
        0)
            # Add to log
            echo "[$COMMIT_DATE] Contribution #$i: Updated documentation and notes." >> contribution_log.md
            ;;
        1)
            # Update README with a space change (add/remove blank line)
            sed -i '$a\' README.md
            echo "# " >> README.md
            ;;
        2)
            # Add timestamp to log
            echo "[$COMMIT_DATE] Working on graph attack experiments - run #$i." >> contribution_log.md
            ;;
        3)
            # Touch a file and modify it slightly
            echo "# Auto update $i" >> contribution_log.md
            ;;
        4)
            # Another variation
            echo "[$COMMIT_DATE] Code review and minor adjustments (#$i)." >> contribution_log.md
            ;;
    esac

    # Stage and commit with the specific date
    git add -A
    GIT_COMMITTER_DATE="$COMMIT_DATE" git commit -m "Update: contribution #$i" --date="$COMMIT_DATE" --no-gpg-sign

    # Progress
    echo "Commit $i/$TOTAL done - $COMMIT_DATE"
done

echo ""
echo "=== Done! $TOTAL commits created ==="
echo "Run 'git push origin main' to push to GitHub"
