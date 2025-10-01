"""
Entity Extractor Module
Extracts product names, account numbers, and dates from customer queries
"""

import re
import dateparser
from datetime import datetime
from typing import Dict, List, Any


class EntityExtractor:
    """
    Extracts named entities from customer support queries
    """
    
    def __init__(self):
        # Regex patterns for different entity types
        self.patterns = {
            'account_number': [
                r'account\s*#?\s*(\d{6,12})',
                r'account\s+number\s+(\d{6,12})',
                r'acc\s*#?\s*(\d{6,12})',
                r'\b(\d{10,12})\b'  # standalone account numbers
            ],
            'product_name': [
                r'product\s+([A-Za-z0-9\-]+)',
                r'([A-Za-z0-9\-]+)\s+product',
                r'with\s+([A-Za-z][A-Za-z0-9\-]+)',
                r'for\s+([A-Za-z][A-Za-z0-9\-]+)',
            ],
            'order_number': [
                r'order\s*#?\s*([A-Z0-9]{6,15})',
                r'order\s+number\s+([A-Z0-9]{6,15})',
                r'order\s+id\s+([A-Z0-9]{6,15})'
            ],
            'amount': [
                r'\$\s*(\d+(?:\.\d{2})?)',
                r'(\d+(?:\.\d{2})?)\s*dollars?',
                r'amount\s+of\s+\$?(\d+(?:\.\d{2})?)'
            ]
        }
        
        # Common product keywords to filter out false positives
        self.ignore_words = {
            'issue', 'problem', 'help', 'support', 'service', 'account',
            'billing', 'payment', 'refund', 'cancel', 'update', 'change'
        }
    
    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from text
        
        Args:
            text: Input query text
            
        Returns:
            Dictionary containing all extracted entities
        """
        entities = {
            'account_numbers': self.extract_account_numbers(text),
            'product_names': self.extract_product_names(text),
            'order_numbers': self.extract_order_numbers(text),
            'dates': self.extract_dates(text),
            'amounts': self.extract_amounts(text)
        }
        
        return entities
    
    def extract_account_numbers(self, text: str) -> List[str]:
        """Extract account numbers from text"""
        account_numbers = []
        
        for pattern in self.patterns['account_number']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            account_numbers.extend(matches)
        
        # Remove duplicates and return
        return list(set(account_numbers))
    
    def extract_product_names(self, text: str) -> List[str]:
        """Extract product names from text"""
        product_names = []
        
        for pattern in self.patterns['product_name']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Filter out common words that aren't product names
            filtered_matches = [
                m for m in matches 
                if m.lower() not in self.ignore_words and len(m) > 1
            ]
            product_names.extend(filtered_matches)
        
        # Remove duplicates
        return list(set(product_names))
    
    def extract_order_numbers(self, text: str) -> List[str]:
        """Extract order numbers from text"""
        order_numbers = []
        
        for pattern in self.patterns['order_number']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            order_numbers.extend(matches)
        
        return list(set(order_numbers))
    
    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dates from text using dateparser
        
        Returns list of dicts with 'text' and 'parsed_date'
        """
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 12/31/2024 or 12-31-24
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # 2024-12-31
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # January 1, 2024
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',  # 1 January 2024
        ]
        
        # Also look for relative dates
        relative_patterns = [
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\blast\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\bnext\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        ]
        
        all_patterns = date_patterns + relative_patterns
        
        for pattern in all_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(0)
                parsed = dateparser.parse(date_text)
                if parsed:
                    dates.append({
                        'text': date_text,
                        'parsed_date': parsed.strftime('%Y-%m-%d'),
                        'datetime': parsed
                    })
        
        return dates
    
    def extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts from text"""
        amounts = []
        
        for pattern in self.patterns['amount']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_text = match.group(0)
                amount_value = match.group(1)
                amounts.append({
                    'text': amount_text,
                    'value': float(amount_value)
                })
        
        return amounts
    
    def get_entity_summary(self, text: str) -> str:
        """
        Get a human-readable summary of extracted entities
        
        Args:
            text: Input query text
            
        Returns:
            Formatted string summarizing entities
        """
        entities = self.extract_all(text)
        
        summary_parts = []
        
        if entities['account_numbers']:
            summary_parts.append(f"Account: {', '.join(entities['account_numbers'])}")
        
        if entities['product_names']:
            summary_parts.append(f"Product: {', '.join(entities['product_names'])}")
        
        if entities['order_numbers']:
            summary_parts.append(f"Order: {', '.join(entities['order_numbers'])}")
        
        if entities['dates']:
            date_texts = [d['text'] for d in entities['dates']]
            summary_parts.append(f"Date: {', '.join(date_texts)}")
        
        if entities['amounts']:
            amount_texts = [d['text'] for d in entities['amounts']]
            summary_parts.append(f"Amount: {', '.join(amount_texts)}")
        
        return " | ".join(summary_parts) if summary_parts else "No entities found"


# Example usage and testing
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Test cases
    test_queries = [
        "I have an issue with my bill for product X",
        "My account number is 1234567890 and I need help",
        "Order #ABC123XYZ was charged $49.99 but not delivered",
        "I was charged on January 15, 2024 for account 9876543210",
        "Product Alpha-500 not working, order ORD789456",
        "Refund $29.99 to account #123456789 please"
    ]
    
    print("=" * 60)
    print("ENTITY EXTRACTOR TEST")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        entities = extractor.extract_all(query)
        
        print(f"Account Numbers: {entities['account_numbers']}")
        print(f"Product Names: {entities['product_names']}")
        print(f"Order Numbers: {entities['order_numbers']}")
        print(f"Dates: {[d['text'] for d in entities['dates']]}")
        print(f"Amounts: {[f'{a['text']} (${a['value']})' for a in entities['amounts']]}")
        print(f"\nSummary: {extractor.get_entity_summary(query)}")
        print("=" * 60)