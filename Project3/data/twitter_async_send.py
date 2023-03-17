import tweepy
import json
import asyncio


# tweepy.StreamClient 클래스를 상속받는 클래스
class TwitterStream(tweepy.StreamingClient):
    def __init__(self, bearer_token, user_agent):
        super().__init__(bearer_token)
        self.bearer_token = bearer_token
        self.user_agent = user_agent
        
    def on_data(self, data):
        tweet = json.loads(data)
        if '@' not in tweet['data']['text'] and tweet['data']['lang'] == 'en':
            print(tweet)
            return tweet


# 규칙 제거 함수
def delete_all_rules(client, rules):
    # 규칙 값이 없는 경우 None으로 들어옴
    if rules is None or rules.data is None:
        return None
    stream_rules = rules.data
    client.delete_rules(ids=list(map(lambda rule: rule.id, stream_rules)))


# 스트리밍
def send_data(keyword):
    print("Start getting data from Twitter")
    
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPLxggEAAAAAEuPi0BJKhaGD3OtAzB%2BfEq6MPHQ%3DNMDgsGP2POmNQn0hghqw23Ilgz134UjZXfYHCpbaU2QYXxtwG2"
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
    
    client = TwitterStream(BEARER_TOKEN, USER_AGENT)    # 스트림 클라이언트 인스턴스 생성
    delete_all_rules(client, client.get_rules())    # 모든 규칙 제거
    
    client.add_rules(tweepy.StreamRule(value=keyword))          # 스트림 규칙 추가
    tweet = client.filter(tweet_fields=['author_id', 'lang', 'created_at', 'public_metrics'])    # 스트림 시작
    print(tweet)

    return tweet



async def handle_echo(reader, writer):
    data = send_data('#football')
    writer.write(data.encode())
    await writer.drain()
    writer.close()

    
async def main():
    server = await asyncio.start_server(
        handle_echo, '127.0.0.1', 5555)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

        
asyncio.run(main())