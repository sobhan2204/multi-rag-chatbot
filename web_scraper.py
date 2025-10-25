import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

class PDFScraper:
    def __init__(self, base_url, data_folder='data'):
        """
        Initialize the PDF scraper
        
        Args:
            base_url: The website URL to scrape
            data_folder: Folder to save PDFs (default: 'data')
        """
        self.base_url = base_url
        self.data_folder = data_folder
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_pdf_links(self, url=None):
        """
        Extract all PDF links from a webpage
        
        Args:
            url: URL to scrape (uses base_url if not provided)
        
        Returns:
            List of PDF URLs
        """
        if url is None:
            url = self.base_url
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_links = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Check if link points to PDF
                if href.lower().endswith('.pdf'):
                    full_url = urljoin(url, href)
                    pdf_links.append(full_url)
            
            print(f"Found {len(pdf_links)} PDF links")
            return pdf_links
        
        except Exception as e:
            print(f"Error fetching page: {e}")
            return []
    
    def download_pdf(self, pdf_url, custom_name=None):
        """
        Download a single PDF
        
        Args:
            pdf_url: URL of the PDF
            custom_name: Optional custom filename
        
        Returns:
            Path to saved file or None if failed
        """
        try:
            print(f"Downloading: {pdf_url}")
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Generate filename
            if custom_name:
                filename = custom_name if custom_name.endswith('.pdf') else f"{custom_name}.pdf"
            else:
                filename = os.path.basename(urlparse(pdf_url).path)
                if not filename:
                    filename = f"document_{int(time.time())}.pdf"
            
            filepath = os.path.join(self.data_folder, filename)
            
            # Save PDF
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Saved: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
            return None
    
    def scrape_all_pdfs(self, delay=1):
        """
        Scrape and download all PDFs from the base URL
        
        Args:
            delay: Delay between downloads in seconds (default: 1)
        
        Returns:
            List of downloaded file paths
        """
        pdf_links = self.get_pdf_links()
        downloaded_files = []
        
        for i, pdf_url in enumerate(pdf_links, 1):
            print(f"\nDownloading PDF {i}/{len(pdf_links)}")
            filepath = self.download_pdf(pdf_url)
            
            if filepath:
                downloaded_files.append(filepath)
            
            # Be polite to the server
            if i < len(pdf_links):
                time.sleep(delay)
        
        print(f"\n{'='*50}")
        print(f"Download complete: {len(downloaded_files)}/{len(pdf_links)} PDFs saved to '{self.data_folder}'")
        return downloaded_files


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PDF Web Scraper for RAG Pipeline")
    print("="*60)
    
    # Get URL from user
    website_url = input("\nEnter the website URL to scrape PDFs from: ").strip()
    
    if not website_url:
        print("Error: URL cannot be empty!")
        exit(1)
    
    # Add https:// if not present
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    print(f"\nTarget URL: {website_url}")
    print(f"Save location: data/")
    print("\nStarting scraper...\n")
    
    # Initialize scraper - always use 'data' folder
    scraper = PDFScraper(website_url, data_folder='data')
    
    # Scrape all PDFs
    downloaded_files = scraper.scrape_all_pdfs(delay=1)
    
    # Show results
    if downloaded_files:
        print(f"\n✓ Successfully downloaded {len(downloaded_files)} PDFs:")
        for filepath in downloaded_files:
            print(f"  - {filepath}")
    else:
        print("\n✗ No PDFs were downloaded. Please check:")
        print("  1. The URL is correct")
        print("  2. The page contains PDF links")
        print("  3. You have internet connection")