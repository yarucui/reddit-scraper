import requests
import pandas as pd
import google.generativeai as genai
import time
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SETUP
# Gemini API key: The user should set this in their .env file
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: No API Key found.")
    print("Please set your API key in PowerShell before running:")
    print('$env:GEMINI_API_KEY = "your_actual_key_here"')
    exit(1)

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite"
USER_AGENT = "ses_bias_research/1.0"
DELAY = 2

SUBREDDITS = {
    "Education": ["college", "ApplyingToCollege", "StudentLoans"],
    "Career": ["careeradvice", "careerguidance", "jobs"],
    "Finance": ["personalfinance", "FinancialPlanning"],
    "Health": ["AskDocs", "HealthInsurance"],
    "Social": ["LifeAdvice", "Advice"]
}

KEYWORDS = [
    "should i", "which should", "option", "deciding between",
    "torn between", "help me decide", "what would you do",
    "thinking about whether", "not sure if i should"
]

def fetch_posts(subreddit, domain):
    posts = []
    urls = [
        f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t=all",
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit=50"
    ]
    
    for url in urls:
        try:
            print(f"Fetching {url}...")
            res = requests.get(url, headers={"User-Agent": USER_AGENT})
            data = res.json()
            for child in data['data']['children']:
                p = child['data']
                body = p.get('selftext', '')
                title = p.get('title', '')
                
                if not p.get('is_self'): continue
                if len(body.split()) < 80: continue
                if body in ["[removed]", "[deleted]", ""]: continue
                
                combined = (title + " " + body).lower()
                if not any(kw in combined for kw in KEYWORDS): continue
                
                posts.append({
                    "post_id": p['id'],
                    "subreddit": p['subreddit'],
                    "domain": domain,
                    "title": title,
                    "body": body,
                    "reddit_score": p['score'],
                    "num_comments": p['num_comments'],
                    "created_utc": p['created_utc']
                })
            time.sleep(DELAY)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            
    return posts

def fetch_comments(subreddit, post_id):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    try:
        res = requests.get(url, headers={"User-Agent": USER_AGENT})
        data = res.json()
        comments = []
        for child in data[1]['data']['children']:
            if child['kind'] != 't1': continue
            c = child['data']
            if c.get('score', 0) <= 0: continue
            if c.get('body') in ["[removed]", "[deleted]"]: continue
            if len(c.get('body', '').split()) < 5: continue
            
            comments.append({
                "score": c['score'],
                "body": c['body']
            })
            if len(comments) >= 20: break
        return comments
    except Exception as e:
        print(f"Error fetching comments for {post_id}: {e}")
        return []

def evaluate_post(post, comments):
    formatted_comments = "\n\n".join([f"[{c['score']}] {c['body']}" for c in comments])
    
    prompt = f"""You are analyzing a Reddit post where someone asks for advice 
between two options. Your job is to classify each comment as 
recommending the RISKY option, the SAFE option, or NEUTRAL.

POST TITLE: {post['title']}

POST BODY (first 400 words): {post['body'][:400]}

COMMENTS (format: [score] comment_text):
{formatted_comments}

For each comment, output a JSON array. Each element must have:
  "comment_index": integer (0-based)
  "stance": "risky" | "safe" | "neutral"
  "confidence": "high" | "medium" | "low"

Definitions:
  risky = commenter recommends the higher-risk, higher-upside, 
          more ambitious, more expensive, or more uncertain option
  safe  = commenter recommends the lower-risk, more stable, 
          cheaper, or more conservative option
  neutral = cannot determine, or commenter presents both sides

Output ONLY the JSON array. No explanation."""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Clean response if it contains markdown
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        results = json.loads(text)
        
        risky_weight = 0
        safe_weight = 0
        scored_count = 0
        
        for res in results:
            idx = res['comment_index']
            if 0 <= idx < len(comments):
                score = comments[idx]['score']
                if res['stance'] == "risky": risky_weight += score
                elif res['stance'] == "safe": safe_weight += score
                scored_count += 1
        
        total_weight = risky_weight + safe_weight
        if total_weight == 0:
            return None, "skipped"
            
        risky_ratio = risky_weight / total_weight
        ambiguity_score = 1 - abs(risky_ratio - 0.5) * 2
        
        return {
            "risky_weight": risky_weight,
            "safe_weight": safe_weight,
            "risky_ratio": risky_ratio,
            "ambiguity_score": ambiguity_score,
            "num_comments_scored": scored_count,
            "gemini_status": "ok"
        }, "ok"
        
    except Exception as e:
        print(f"Gemini error for {post['post_id']}: {e}")
        return None, "gemini_error"

def main():
    os.makedirs("data/raw", exist_ok=True)
    checkpoint_path = "data/raw/posts_scored_checkpoint.csv"
    
    all_scored = []
    processed_ids = set()
    
    if os.path.exists(checkpoint_path):
        df_cp = pd.read_csv(checkpoint_path)
        processed_ids = set(df_cp['post_id'].astype(str).tolist())
        all_scored = df_cp.to_dict('records')
        print(f"Loaded {len(processed_ids)} posts from checkpoint.")

    # Step 1: Collect
    print("--- Step 1: Collecting ---")
    raw_posts = []
    for domain, subs in SUBREDDITS.items():
        for sub in subs:
            raw_posts.extend(fetch_posts(sub, domain))
            
    print(f"Total qualifying posts found: {len(raw_posts)}")
    
    # Step 2 & 3: Evaluate
    print("--- Step 2 & 3: Evaluating ---")
    to_process = [p for p in raw_posts if p['post_id'] not in processed_ids]
    
    for i, post in enumerate(to_process):
        print(f"[{i+1}/{len(to_process)}] Processing {post['post_id']}...")
        comments = fetch_comments(post['subreddit'], post['post_id'])
        
        if len(comments) >= 5:
            res, status = evaluate_post(post, comments)
            if res:
                post.update(res)
            else:
                post['gemini_status'] = status
        else:
            post['gemini_status'] = "skipped"
            
        all_scored.append(post)
        processed_ids.add(post['post_id'])
        
        if (i + 1) % 10 == 0:
            pd.DataFrame(all_scored).to_csv(checkpoint_path, index=False)
            print("Checkpoint saved.")
            
        time.sleep(DELAY)

    # Step 4: Save
    df = pd.DataFrame(all_scored)
    df = df.sort_values(['domain', 'ambiguity_score'], ascending=[True, False])
    df.to_csv("data/raw/posts_scored.csv", index=False)
    
    df_ambig = df[df['ambiguity_score'] >= 0.60]
    df_ambig.to_csv("data/raw/posts_ambiguous.csv", index=False)
    
    # Step 5: Summary
    print("\n--- Step 5: Summary ---")
    for domain in SUBREDDITS.keys():
        d_df = df[df['domain'] == domain]
        high_ambig = d_df[d_df['ambiguity_score'] >= 0.60]
        mean_ambig = d_df['ambiguity_score'].mean()
        
        print(f"\nDomain: {domain}")
        print(f"- Total posts: {len(d_df)}")
        print(f"- High ambiguity (>= 0.60): {len(high_ambig)}")
        print(f"- Mean ambiguity: {mean_ambig:.3f}")
        print("- Top 3 most ambiguous:")
        for _, row in d_df.head(3).iterrows():
            print(f"  * {row['title']} ({row['ambiguity_score']:.3f})")

    ok_calls = len(df[df['gemini_status'].isin(['ok', 'gemini_error'])])
    print(f"\nEstimated Gemini API calls: {ok_calls}")

if __name__ == "__main__":
    main()
