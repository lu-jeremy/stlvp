git filter-branch -f --env-filter '
WRONG_EMAIL="vidyaak2903@gmail.com"
NEW_NAME="Jeremy Lu"
NEW_EMAIL="lu1008@purdue.edu"

if [ "$GIT_COMMITTER_EMAIL" = "$WRONG_EMAIL" ]
then
	export GIT_COMMITTER_NAME="$NEW_NAME"
	export GIT_COMMITTER_EMAIL="$NEW_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$WRONG_EMAIL" ]
then
	export GIT_AUTHOR_NAME="$NEW_NAME"
	export GIT_AUTHOR_EMAIL="$NEW_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags
