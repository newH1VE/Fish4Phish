

# entry containing all features for content filter


class FeatureEntryContent:

    def __init__(self, bool_redirect_website, bool_favicon_website, bool_content_extern_website, int_links_extern_website, bool_custom_statusbar_website,
                 bool_disable_rightclick_website, bool_popup_website, bool_iframe_website, bool_action_website, bool_action_extern_website,
                 bool_form_post_website, int_phishy_tokens_website, bool_input_website, float_description_sim_website, bool_bond_status_website,
                 bool_freq_domain_extern_website, float_login_home_website, bool_copyright_website, float_copyright_sim_website,
                 float_title_sim_website, float_unique_links_website, int_input_website,
                 bool_input_login_website, bool_button_website, bool_meta_website, bool_hidden_element_website, int_option_website,
                 int_select_website, int_th_website, int_tr_website, int_table_website, int_href_website, int_li_website, int_ul_website,
                 int_ol_website, int_div_website, int_span_website, int_article_website, int_p_website, int_checkbox_website,
                 int_button_website, int_image_website, label, url, final_url):


        self.bool_redirect_website = bool_redirect_website
        self.bool_favicon_website = bool_favicon_website
        self.bool_content_extern_website = bool_content_extern_website
        self.int_links_extern_website = int_links_extern_website
        self.bool_custom_statusbar_website = bool_custom_statusbar_website
        self.bool_disable_rightclick_website = bool_disable_rightclick_website
        self.bool_popup_website = bool_popup_website
        self.bool_iframe_website = bool_iframe_website
        self.bool_action_website = bool_action_website
        self.bool_action_extern_website = bool_action_extern_website
        self.bool_form_post_website = bool_form_post_website
        self.int_phishy_tokens_website = int_phishy_tokens_website
        self.bool_input_website = bool_input_website
        self.float_description_sim_website = float_description_sim_website
        self.bool_bond_status_website = bool_bond_status_website
        self.bool_freq_domain_extern_website = bool_freq_domain_extern_website
        self.float_login_home_website = float_login_home_website
        self.bool_copyright_website = bool_copyright_website
        self.float_copyright_sim_website = float_copyright_sim_website
        self.float_title_sim_website = float_title_sim_website
        self.float_unique_links_website = float_unique_links_website
        #self.bool_link_analysis_website = bool_link_analysis_website
        self.int_input_website = int_input_website
        self.bool_input_login_website = bool_input_login_website
        self.bool_button_website = bool_button_website
        self.bool_meta_website = bool_meta_website
        self.bool_hidden_element_website = bool_hidden_element_website
        self.int_option_website = int_option_website
        self.int_select_website = int_select_website
        self.int_th_website = int_th_website
        self.int_tr_website = int_tr_website
        self.int_table_website = int_table_website
        self.int_href_website = int_href_website
        self.int_li_website = int_li_website
        self.int_ul_website = int_ul_website
        self.int_ol_website = int_ol_website
        self.int_div_website = int_div_website
        self.int_span_website = int_span_website
        self.int_article_website = int_article_website
        self.int_p_website = int_p_website
        self.int_checkbox_website = int_checkbox_website
        self.int_button_website = int_button_website
        self.int_image_website = int_image_website
        self.label = label
        self.url = url
        self.final_url = final_url
