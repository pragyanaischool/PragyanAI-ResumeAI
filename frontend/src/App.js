import React, { useState, useEffect } from 'react';

// Main App component
const App = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [extractedData, setExtractedData] = useState(null);
  const [markdownContent, setMarkdownContent] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [userId, setUserId] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [showExtracted, setShowExtracted] = useState(false);
  const [showMarkdown, setShowMarkdown] = useState(false);
  const [showChat, setShowChat] = useState(false);
  
  // New state for listing and filtering resumes
  const [filteredResumes, setFilteredResumes] = useState([]);
  const [filterQuery, setFilterQuery] = useState('');

  // Determine the backend URL.
  // IMPORTANT: Replace 'https://your-backend-url.onrender.com' with your actual Render URL after deployment.
  const backendUrl = 'https://your-backend-url.onrender.com';

  // Generate a random user ID on component mount
  useEffect(() => {
    // Generate a unique ID for the user
    const uniqueId = crypto.randomUUID();
    setUserId(uniqueId);
  }, []);

  // Handle file selection
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setExtractedData(null);
    setMarkdownContent('');
    setShowExtracted(false);
    setShowMarkdown(false);
  };

  // Upload and process resume
  const processResume = async () => {
    if (!file || !userId) {
      setStatusMessage('Please select a file and ensure you are authenticated.');
      return;
    }
    setLoading(true);
    setStatusMessage('Processing resume...');
    const formData = new FormData();
    formData.append('resume_file', file);
    formData.append('user_id', userId);

    try {
      const response = await fetch(`${backendUrl}/api/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setExtractedData(result.extracted_data);
      setMarkdownContent(result.markdown_content);
      setStatusMessage('Resume processed and saved successfully!');
      setShowExtracted(true);
      setShowMarkdown(true);
      
    } catch (error) {
      console.error('Error processing resume:', error);
      setStatusMessage(`Error: ${error.message}. Please check the console and ensure the backend is running.`);
    } finally {
      setLoading(false);
    }
  };

  // Handle chat query
  const handleChat = async () => {
    if (!chatInput) return;
    setLoading(true);
    setChatHistory(prev => [...prev, { sender: 'user', text: chatInput }]);
    const currentChat = chatInput;
    setChatInput('');
    setStatusMessage('Getting response...');

    try {
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentChat }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setChatHistory(prev => [...prev, { sender: 'ai', text: result.response }]);
      setStatusMessage('Response received.');

    } catch (error) {
      console.error('Error with chat:', error);
      setChatHistory(prev => [...prev, { sender: 'ai', text: `Error: ${error.message}` }]);
      setStatusMessage('Chat error. See console for details.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle listing all resumes
  const handleListAllResumes = async () => {
    setLoading(true);
    setStatusMessage('Fetching all resumes...');
    try {
      const response = await fetch(`${backendUrl}/api/resumes/list`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFilteredResumes(data.resumes);
      setStatusMessage('All resumes fetched successfully!');
    } catch (error) {
      console.error('Error fetching resumes:', error);
      setStatusMessage(`Error fetching resumes: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle filtering resumes based on query
  const handleFilterResumes = async () => {
    if (!filterQuery) return;
    setLoading(true);
    setStatusMessage('Filtering resumes...');
    try {
      const response = await fetch(`${backendUrl}/api/resumes/filter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: filterQuery }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setFilteredResumes(data.resumes);
      setStatusMessage('Resumes filtered successfully!');
    } catch (error) {
      console.error('Error filtering resumes:', error);
      setStatusMessage(`Error filtering resumes: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 min-h-screen text-gray-100 font-sans p-8 flex flex-col items-center">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .font-inter { font-family: 'Inter', sans-serif; }
      `}</style>
      <div className="w-full max-w-6xl space-y-8 font-inter">
        <header className="text-center">
          <h1 className="text-5xl font-extrabold text-blue-400 mb-2">Resume AI</h1>
          <p className="text-lg text-gray-400">Upload, analyze, and query your resumes.</p>
        </header>

        <section className="bg-gray-800 p-8 rounded-2xl shadow-lg flex flex-col items-center space-y-6">
          <div className="w-full max-w-md space-y-4">
            <h2 className="text-3xl font-bold text-gray-200 text-center">Upload Resume</h2>
            <p className="text-gray-400 text-center">
              Upload a `.txt`, `.pdf`, `.doc`, or `.docx` file to extract key information, generate a professional markdown version, and add it to the RAG model.
            </p>
            <label htmlFor="resume-upload" className="flex justify-center items-center w-full px-4 py-3 bg-blue-500 hover:bg-blue-600 transition-colors duration-200 rounded-xl cursor-pointer text-white font-semibold shadow-md">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 5-4V15z" clipRule="evenodd"></path></svg>
              {file ? file.name : 'Choose a Resume File (.txt, .pdf, .doc, .docx)'}
              <input id="resume-upload" type="file" className="hidden" onChange={handleFileChange} accept=".txt,.pdf,.doc,.docx" />
            </label>
            <button onClick={processResume} disabled={loading || !file} className="w-full px-4 py-3 bg-green-500 hover:bg-green-600 transition-colors duration-200 rounded-xl text-white font-semibold shadow-md disabled:bg-gray-600 disabled:cursor-not-allowed">
              {loading ? 'Processing...' : 'Process & Save Resume'}
            </button>
          </div>
          <p className="text-center text-gray-400 min-h-[2rem]">{statusMessage}</p>
        </section>

        {showExtracted && extractedData && (
          <section className="bg-gray-800 p-8 rounded-2xl shadow-lg space-y-4">
            <h2 className="text-3xl font-bold text-gray-200 text-center">Extracted Data</h2>
            <div className="bg-gray-700 text-gray-300 p-6 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono text-sm">
              {JSON.stringify(extractedData, null, 2)}
            </div>
          </section>
        )}

        {showMarkdown && markdownContent && (
          <section className="bg-gray-800 p-8 rounded-2xl shadow-lg space-y-4">
            <h2 className="text-3xl font-bold text-gray-200 text-center">Markdown Resume</h2>
            <div className="bg-gray-700 text-gray-300 p-6 rounded-xl markdown">
              <div dangerouslySetInnerHTML={{ __html: marked.parse(markdownContent) }} />
            </div>
            <p className="text-sm text-gray-500 text-center">
              The markdown content is a beautifully formatted version of your resume. You can copy it to use anywhere.
            </p>
          </section>
        )}

        <section className="bg-gray-800 p-8 rounded-2xl shadow-lg space-y-4">
          <h2 className="text-3xl font-bold text-gray-200 text-center">Ask the AI about Resumes</h2>
          <p className="text-gray-400 text-center">
            Query the processed resumes. Example: "List all resumes with experience in Python and Machine Learning."
          </p>
          <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
            <div className="flex-1 bg-gray-700 p-4 rounded-xl max-h-96 overflow-y-auto space-y-4 custom-scrollbar">
              {chatHistory.length === 0 ? (
                <div className="text-center text-gray-500 italic">Chat history will appear here.</div>
              ) : (
                chatHistory.map((msg, index) => (
                  <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`p-4 rounded-2xl max-w-[80%] ${msg.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-200'}`}>
                      {msg.text}
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="flex flex-col space-y-2 w-full md:w-1/3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                placeholder="Ask a question..."
                className="w-full p-3 rounded-xl bg-gray-700 border border-gray-600 text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <button onClick={handleChat} disabled={loading || !chatInput} className="w-full px-4 py-3 bg-blue-500 hover:bg-blue-600 transition-colors duration-200 rounded-xl text-white font-semibold disabled:bg-gray-600 disabled:cursor-not-allowed">
                {loading ? 'Thinking...' : 'Send'}
              </button>
            </div>
          </div>
        </section>
        
        {/* New section for listing and filtering resumes */}
        <section className="bg-gray-800 p-8 rounded-2xl shadow-lg space-y-4">
          <h2 className="text-3xl font-bold text-gray-200 text-center">Filter Resumes</h2>
          <p className="text-gray-400 text-center">
            List all resumes or use the AI to filter them based on skills, education, or other criteria. Example: "skills: Python, SQL" or "education: Computer Science".
          </p>
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
            <button onClick={handleListAllResumes} className="flex-1 px-4 py-3 bg-indigo-500 hover:bg-indigo-600 transition-colors duration-200 rounded-xl text-white font-semibold disabled:bg-gray-600 disabled:cursor-not-allowed">
              List All Resumes
            </button>
            <div className="flex-1 flex flex-col space-y-2">
              <input
                type="text"
                value={filterQuery}
                onChange={(e) => setFilterQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleFilterResumes()}
                placeholder="e.g., skills: Python, SQL"
                className="w-full p-3 rounded-xl bg-gray-700 border border-gray-600 text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                disabled={loading}
              />
              <button onClick={handleFilterResumes} disabled={loading || !filterQuery} className="w-full px-4 py-3 bg-indigo-500 hover:bg-indigo-600 transition-colors duration-200 rounded-xl text-white font-semibold disabled:bg-gray-600 disabled:cursor-not-allowed">
                Filter Resumes
              </button>
            </div>
          </div>
          
          <div className="mt-6 space-y-4">
            {filteredResumes.length > 0 ? (
              filteredResumes.map(resume => (
                <div key={resume.id} className="bg-gray-700 p-4 rounded-xl shadow-inner">
                  <h3 className="text-xl font-bold text-blue-400">{resume.extracted_data?.name || "Unknown"}</h3>
                  <p className="text-sm text-gray-400 italic">ID: {resume.id}</p>
                  <p className="text-sm mt-2 text-gray-300">Skills: {(resume.extracted_data?.skills || []).join(', ')}</p>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-500 italic">No resumes to display. Upload a resume first or use the filter function.</p>
            )}
          </div>
        </section>

        <p className="text-center text-xs text-gray-500">
          Your User ID is: {userId || 'Authenticating...'}
        </p>
      </div>
      <script src="https://cdn.tailwindcss.com"></script>
      <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </div>
  );
};

export default App;
