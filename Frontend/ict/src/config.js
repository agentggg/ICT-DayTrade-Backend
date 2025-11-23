
const ipAddress = {
  api_endpoint: 'http://10.0.0.159:8000',
//   api_endpoint: 'http://127.0.0.1:8000'
//   api_endpoint: 'https://allnations-agentofgod.pythonanywhere.com'
}

 
const getIpAddress = () => {
    return ipAddress.api_endpoint
}


export default getIpAddress