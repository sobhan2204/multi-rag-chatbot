import React, { useState, useEffect } from 'react';
import { Send, Globe, FileText, Loader2, CheckCircle, AlertCircle, FolderOpen } from 'lucide-react';

export default function MultiRAGChatbot() {
  const [activeTab, setActiveTab] = useState('scraper');
  const [scrapingUrl, setScrapingUrl] = useState('');
  const [chatQuery, setChatQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [scrapingStatus, setScrapingStatus] = useState('idle');
  const [scrapingMessage, setScrapingMessage] = useState('');
  const [dataFiles, setDataFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  // Simulate loading files from data folder
  useEffect(() => {
    loadDataFiles();
  }, []);

  const loadDataFiles = async () => {
    try {
      // This will call your backend API to list files in data folder
      const response = await fetch('/api/list-files');
      const data = await response.json();
      setDataFiles(data.files || []);
    } catch (error) {
      // Fallback demo data
      setDataFiles([
        { name: 'insurance_policy.pdf', size: '2.4 MB', date: '2024-11-10' },
        { name: 'terms_conditions.pdf', size: '1.8 MB', date: '2024-11-09' },
        { name: 'claims_guide.pdf', size: '3.1 MB', date: '2024-11-08' }
      ]);
    }
  };

  const handleScraping = async () => {
    if (!scrapingUrl.trim()) {
      setScrapingMessage('Please enter a valid URL');
      setScrapingStatus('error');
      return;
    }

    setScrapingStatus('loading');
    setScrapingMessage('Scraping PDFs from website...');

    try {
      const response = await fetch('/api/scrape-pdfs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: scrapingUrl })
      });

      const data = await response.json();

      if (data.success) {
        setScrapingStatus('success');
        setScrapingMessage(`Successfully downloaded ${data.count} PDFs`);
        loadDataFiles(); // Refresh file list
        setScrapingUrl('');
      } else {
        setScrapingStatus('error');
        setScrapingMessage(data.message || 'Failed to scrape PDFs');
      }
    } catch (error) {
      setScrapingStatus('error');
      setScrapingMessage('Error connecting to server');
    }
  };

  const handleChatSubmit = async () => {
    if (!chatQuery.trim()) return;

    const userMessage = { role: 'user', content: chatQuery, timestamp: new Date() };
    setChatHistory(prev => [...prev, userMessage]);
    setChatQuery('');
    setIsProcessing(true);

    try {
      const response = await fetch('/api/query-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: chatQuery,
          show_comparison: showComparison 
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        comparison: data.comparison,
        timestamp: new Date()
      };

      setChatHistory(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Error: Unable to process your query. Please try again.',
        timestamp: new Date()
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/30 backdrop-blur-lg border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
            Multi-RAG Chatbot
          </h1>
          <p className="text-purple-200/70 mt-1">Integrated PDF Scraping & Intelligent Query System</p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-6 pt-6">
        <div className="flex gap-2 bg-black/30 p-1 rounded-lg backdrop-blur-lg border border-purple-500/20 w-fit">
          {[
            { id: 'scraper', label: 'PDF Scraper', icon: Globe },
            { id: 'chat', label: 'RAG Chat', icon: Send },
            { id: 'files', label: 'Data Files', icon: FolderOpen }
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-md transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                    : 'text-purple-200/70 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* PDF Scraper Tab */}
        {activeTab === 'scraper' && (
          <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-purple-500/20 p-8">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Globe className="text-purple-400" />
              PDF Web Scraper
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-purple-200 mb-2 font-medium">Website URL</label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={scrapingUrl}
                    onChange={(e) => setScrapingUrl(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleScraping()}
                    placeholder="https://example.com"
                    className="flex-1 bg-black/40 border border-purple-500/30 rounded-lg px-4 py-3 text-white placeholder-purple-300/30 focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20"
                    disabled={scrapingStatus === 'loading'}
                  />
                  <button
                    onClick={handleScraping}
                    disabled={scrapingStatus === 'loading'}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-3 rounded-lg font-medium hover:from-purple-600 hover:to-pink-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-purple-500/30"
                  >
                    {scrapingStatus === 'loading' ? (
                      <>
                        <Loader2 size={18} className="animate-spin" />
                        Scraping...
                      </>
                    ) : (
                      <>
                        <Globe size={18} />
                        Scrape PDFs
                      </>
                    )}
                  </button>
                </div>
              </div>

              {scrapingMessage && (
                <div className={`p-4 rounded-lg flex items-start gap-3 ${
                  scrapingStatus === 'success' ? 'bg-green-500/10 border border-green-500/30' :
                  scrapingStatus === 'error' ? 'bg-red-500/10 border border-red-500/30' :
                  'bg-purple-500/10 border border-purple-500/30'
                }`}>
                  {scrapingStatus === 'success' && <CheckCircle className="text-green-400 flex-shrink-0 mt-0.5" size={20} />}
                  {scrapingStatus === 'error' && <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />}
                  {scrapingStatus === 'loading' && <Loader2 className="text-purple-400 flex-shrink-0 mt-0.5 animate-spin" size={20} />}
                  <p className={
                    scrapingStatus === 'success' ? 'text-green-200' :
                    scrapingStatus === 'error' ? 'text-red-200' :
                    'text-purple-200'
                  }>{scrapingMessage}</p>
                </div>
              )}

              <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-4 mt-6">
                <h3 className="text-purple-200 font-medium mb-2">How it works:</h3>
                <ul className="text-purple-300/70 space-y-1 text-sm">
                  <li>• Enter a website URL containing PDF documents</li>
                  <li>• The scraper will automatically find and download all PDFs</li>
                  <li>• Downloaded files are saved to the 'data' folder</li>
                  <li>• Use the RAG Chat to query the downloaded documents</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* RAG Chat Tab */}
        {activeTab === 'chat' && (
          <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-purple-500/20 p-6 h-[calc(100vh-280px)] flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Send className="text-purple-400" />
                RAG Chat
              </h2>
              <label className="flex items-center gap-2 text-purple-200 text-sm">
                <input
                  type="checkbox"
                  checked={showComparison}
                  onChange={(e) => setShowComparison(e.target.checked)}
                  className="w-4 h-4 rounded bg-black/40 border-purple-500/30 text-purple-500 focus:ring-purple-400"
                />
                Show model comparison
              </label>
            </div>

            {/* Chat History */}
            <div className="flex-1 overflow-y-auto mb-4 space-y-4 scrollbar-thin scrollbar-thumb-purple-500/30 scrollbar-track-transparent">
              {chatHistory.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center text-purple-300/50">
                    <Send size={48} className="mx-auto mb-4 opacity-50" />
                    <p>Start a conversation by asking a question about your documents</p>
                  </div>
                </div>
              ) : (
                chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-lg p-4 ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                        : 'bg-black/40 border border-purple-500/20 text-purple-100'
                    }`}>
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                      {msg.comparison && showComparison && (
                        <div className="mt-3 pt-3 border-t border-purple-500/20 text-xs">
                          <p className="text-purple-300 mb-1">Model Comparison:</p>
                          <pre className="text-purple-200/70">{JSON.stringify(msg.comparison, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-black/40 border border-purple-500/20 rounded-lg p-4">
                    <Loader2 className="animate-spin text-purple-400" size={20} />
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="flex gap-3">
              <input
                type="text"
                value={chatQuery}
                onChange={(e) => setChatQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleChatSubmit()}
                placeholder="Ask a question about your documents..."
                className="flex-1 bg-black/40 border border-purple-500/30 rounded-lg px-4 py-3 text-white placeholder-purple-300/30 focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20"
                disabled={isProcessing}
              />
              <button
                onClick={handleChatSubmit}
                disabled={isProcessing || !chatQuery.trim()}
                className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-medium hover:from-purple-600 hover:to-pink-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-purple-500/30"
              >
                {isProcessing ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              </button>
            </div>
          </div>
        )}

        {/* Data Files Tab */}
        {activeTab === 'files' && (
          <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-purple-500/20 p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <FolderOpen className="text-purple-400" />
                Data Files
              </h2>
              <button
                onClick={loadDataFiles}
                className="text-purple-300 hover:text-white transition-colors text-sm flex items-center gap-2"
              >
                <Loader2 size={16} />
                Refresh
              </button>
            </div>

            {dataFiles.length === 0 ? (
              <div className="text-center py-12">
                <FileText size={48} className="mx-auto text-purple-400/30 mb-4" />
                <p className="text-purple-300/50">No files found in data folder</p>
                <p className="text-purple-300/30 text-sm mt-2">Use the PDF Scraper to add documents</p>
              </div>
            ) : (
              <div className="space-y-2">
                {dataFiles.map((file, idx) => (
                  <div key={idx} className="bg-black/40 border border-purple-500/20 rounded-lg p-4 flex items-center justify-between hover:bg-black/60 transition-colors">
                    <div className="flex items-center gap-3">
                      <FileText className="text-purple-400" size={24} />
                      <div>
                        <p className="text-white font-medium">{file.name}</p>
                        <p className="text-purple-300/50 text-sm">{file.size} • {file.date}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}