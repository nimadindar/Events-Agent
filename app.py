import os
import logging
import threading
import streamlit as st

from main import run_agent
from utils.utils import update_agent_config, StreamlitLogHandler


def schedule_agent(every_x_hours, config):
    def run_periodically():
        try:
            logging.info(f"[Scheduler] Running agent (interval: {every_x_hours} hours)")
            update_agent_config(**config)
            run_agent()
        except Exception as e:
            logging.error(f"[Scheduler Error] {e}")
        finally:
            threading.Timer(every_x_hours * 3600, run_periodically).start()

    threading.Timer(0, run_periodically).start()


st.set_page_config(page_title="Research Agent", layout="wide")

st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css');
    
    .main-container {
        @apply max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg;
    }
    .header {
        @apply text-3xl font-bold text-gray-800 mb-6;
    }
    .input-section {
        @apply mb-6;
    }
    .label {
        @apply text-lg font-medium text-gray-700 mb-2;
    }
    .output-section {
        @apply mt-8 p-4 bg-gray-50 rounded-lg;
    }
    .download-button {
        @apply mt-4 inline-block px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700;
    }
    .error-message {
        @apply text-red-600 font-medium mt-4;
    }
    .success-message {
        @apply text-green-600 font-medium mt-4;
    }
    </style>
""", unsafe_allow_html=True)


def main():

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="header">Research Agent Interface</h1>', unsafe_allow_html=True)

    if 'log_handler' not in st.session_state:
        st.session_state.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(st.session_state.log_handler)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    st.markdown('<p class="label">Research Field</p>', unsafe_allow_html=True)
    field = st.text_input("", value="Spatio Temporal Point Process", placeholder="Enter research field")

    st.markdown('<p class="label">Model Name</p>', unsafe_allow_html=True)
    model_name = st.selectbox("", [
        "gemini-2.5-pro", "gemini-2.5-flash","gemini-2.0-flash"
    ], index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="label">Arxiv Max Results</p>', unsafe_allow_html=True)
        arxiv_max_results = st.number_input("", min_value=1, max_value=10, value=5, step=1, key="arxiv_max_results")
    with col2:
        st.markdown('<p class="label">Tavily Max Results</p>', unsafe_allow_html=True)
        tavily_max_results = st.number_input("", min_value=1, max_value=10, value=5, step=1, key="tavily_max_results")

    st.markdown('<p class="label">Temperature</p>', unsafe_allow_html=True)
    temperature = st.slider("", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown('<p class="label">Verbose Logging</p>', unsafe_allow_html=True)
    verbose = st.checkbox("Enable verbose logging", value=True)

    scheduler_enabled = st.checkbox("Enable Scheduler", key="scheduler_enabled")
    scheduler_hours = None
    if scheduler_enabled:
        st.markdown('<p class="label">Run Agent Every (hours)</p>', unsafe_allow_html=True)
        scheduler_hours = st.number_input("Run Interval (hours)", min_value=1, max_value=24, step=1, key="scheduler_hours")

    st.markdown('<p class="label">API Keys (Required)</p>', unsafe_allow_html=True)
    with st.expander("🔐 Set API Keys", expanded=True):
        required_keys = [
            "GOOGLE_API_KEY", "X_API_KEY", "X_API_KEY_SECRET",
            "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET","TAVILY_API_KEY"
        ]
        for key in required_keys:
            env_value = os.getenv(key)
            if key not in st.session_state:
                st.session_state[key] = env_value or ""
                # st.session_state[key] = ""
            st.session_state[key] = st.text_input(
                key.replace("_", " ").title(), value=st.session_state[key], type="password"
            )

    if not scheduler_enabled or (scheduler_enabled and scheduler_hours):
        with st.form(key='agent_form'):
            st.markdown('<p class="label">Agent Task</p>', unsafe_allow_html=True)
            invoke_input = st.text_area("", 
                value="Search for the related papers and blog posts and save an organized json file, and finally post a paper or blog post with the highest score to X.", 
                height=100)

            button_label = "Schedule Agent" if scheduler_enabled else "Run Agent"
            submit_button = st.form_submit_button(label=button_label)
    else:
        submit_button = False
        st.info("Please specify how often the agent should run (in hours) to continue.")

    st.markdown('<div class="output-section">', unsafe_allow_html=True)
    st.markdown('<p class="label">Agent Output</p>', unsafe_allow_html=True)

    log_placeholder = st.empty()

    if submit_button:
        st.session_state.log_handler.clear_logs()
        log_placeholder.empty()

        missing_keys = [key for key in required_keys if not st.session_state.get(key)]
        if missing_keys:
            st.markdown(
                f'<p class="error-message">Missing required API keys: {", ".join(missing_keys)}. Please fill in all keys to continue.</p>',
                unsafe_allow_html=True
            )

        try:
            AgentConfig = {
                    "field": field,
                    "arxiv_max_results": int(arxiv_max_results),
                    "tavily_max_results": int(tavily_max_results),
                    "model_name": model_name,
                    "temperature": temperature,
                    "verbose": verbose,
                    "invoke_input": invoke_input,
                    "GOOGLE_API_KEY": st.session_state.GOOGLE_API_KEY,
                    "X_API_KEY": st.session_state.X_API_KEY,
                    "X_API_KEY_SECRET": st.session_state.X_API_KEY_SECRET,
                    "X_ACCESS_TOKEN": st.session_state.X_ACCESS_TOKEN,
                    "X_ACCESS_TOKEN_SECRET": st.session_state.X_ACCESS_TOKEN_SECRET,
                    "TAVILY_API_KEY": st.session_state.TAVILY_API_KEY
                }

            if scheduler_enabled:
                schedule_agent(scheduler_hours, AgentConfig)
                st.markdown(f'<p class="success-message">Scheduler started. Agent will run every {scheduler_hours} hour(s).</p>', unsafe_allow_html=True)
            else:
                with st.spinner("Running agent..."):
                    update_agent_config(**AgentConfig)
                    print(AgentConfig["TAVILY_API_KEY"])
                    result = run_agent()

                output = result.get('output', 'No output returned')
                st.markdown(f'<p>{output}</p>', unsafe_allow_html=True)

                if verbose:
                    logs = st.session_state.log_handler.get_logs()
                    if logs:
                        st.markdown('<p class="label">Verbose Logs</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="log-output">{logs}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p>No verbose logs captured.</p>', unsafe_allow_html=True)

                st.markdown('<p class="success-message">Agent execution completed successfully!</p>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<p class="error-message">Error running agent: {str(e)}</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # output-section
    st.markdown('</div>', unsafe_allow_html=True)  # main-container


if __name__ == "__main__":
    main()

