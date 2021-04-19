


# entry containing all features for lexical filter

class FeatureEntryLexical:

    def __init__(self, bool_ip_netloc, int_length_url, bool_redirect_url, bool_at_symbol_netloc, bool_prefix_suffix_netloc, bool_subdomain_netloc,int_subdomain_netloc, bool_https_protocol_netloc,
                 bool_other_ports_netloc, bool_https_token_url, int_redirect_url, float_cap_noncap_letters_url, int_dots_url, int_length_netloc, int_domains_netloc,
                 int_dash_netloc, int_domain_tokens_netloc, int_digits_netloc, int_digits_path, int_phishy_tokens_netloc, int_phishy_tokens_path, bool_brand_subdomain,
                 bool_brand_path, bool_query_url, int_query_values_url, int_dots_netloc, int_underscores_netloc, bool_validate_tld_url, int_slash_path, int_comma_url, int_star_url, int_semicolon_url,
                 int_plus_url, bool_javascript_url, int_equals_url, int_dash_url, bool_fragment_url, int_fragment_values_url, int_ampersand_url, bool_html_url, int_tilde_url, int_symbols_url, float_entropy_url,
                 float_vowel_consonant_url, bool_digits_netloc, float_digits_letters_url, int_dash_path, bool_domain_restlive_host, bool_created_shortly_host, float_percent_netloc_url,
                 float_percent_path_url, float_percent_query_url, float_percent_fragment_url, float_divergence_url, bool_shortening_url, label, url, final_url):


        self.bool_ip_netloc = bool_ip_netloc
        self.int_length_url = int_length_url
        self.bool_redirect_url = bool_redirect_url
        self.bool_at_symbol_netloc = bool_at_symbol_netloc
        self.bool_prefix_suffix_netloc = bool_prefix_suffix_netloc
        self.bool_subdomain_netloc = bool_subdomain_netloc
        self.int_subdomain_netloc = int_subdomain_netloc
        self.bool_https_protocol_netloc = bool_https_protocol_netloc
        self.bool_other_ports_netloc = bool_other_ports_netloc
        self.bool_https_token_url = bool_https_token_url
        self.int_redirect_url = int_redirect_url
        self.float_cap_noncap_letters_url = float_cap_noncap_letters_url
        self.int_dots_url = int_dots_url
        self.int_length_netloc = int_length_netloc
        self.int_domains_netloc = int_domains_netloc
        self.int_dash_netloc = int_dash_netloc
        self.int_domain_tokens_netloc = int_domain_tokens_netloc
        self.int_digits_netloc = int_digits_netloc
        self.int_digits_path = int_digits_path
        self.int_phishy_tokens_netloc = int_phishy_tokens_netloc
        self.int_phishy_tokens_path = int_phishy_tokens_path
        self.bool_brand_subdomain = bool_brand_subdomain
        self.bool_brand_path = bool_brand_path
        self.bool_query_url = bool_query_url
        self.int_query_values_url = int_query_values_url
        self.int_dots_netloc = int_dots_netloc
        self.int_underscores_netloc = int_underscores_netloc
        self.bool_validate_tld_url = bool_validate_tld_url
        self.int_slash_path = int_slash_path
        self.int_comma_url = int_comma_url
        self.int_star_url = int_star_url
        self.int_semicolon_url = int_semicolon_url
        self.int_plus_url = int_plus_url
        self.bool_javascript_url = bool_javascript_url
        self.int_equals_url = int_equals_url
        self.int_dash_url = int_dash_url
        self.bool_fragment_url = bool_fragment_url
        self.int_fragment_values_url = int_fragment_values_url
        self.int_ampersand_url = int_ampersand_url
        self.bool_html_url = bool_html_url
        self.int_tilde_url = int_tilde_url
        self.int_symbols_url = int_symbols_url
        self.float_entropy_url = float_entropy_url
        self.float_vowel_consonant_url = float_vowel_consonant_url
        self.bool_digits_netloc = bool_digits_netloc
        self.float_digits_letters_url = float_digits_letters_url
        self.int_dash_path = int_dash_path
        self.bool_domain_restlive_host = bool_domain_restlive_host
        self.bool_created_shortly_host = bool_created_shortly_host
        self.float_percent_netloc_url = float_percent_netloc_url
        self.float_percent_path_url = float_percent_path_url
        self.float_percent_query_url = float_percent_query_url
        self.float_percent_fragment_url = float_percent_fragment_url
        self.float_divergence_url = float_divergence_url
        self.bool_shortening_url = bool_shortening_url
        self.label = label
        self.url = url
        self.final_url = final_url