import base64
import requests
import json
import urllib.parse
import os
import time
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
import threading
import sys
import concurrent.futures
from discord_webhook import DiscordWebhook, DiscordEmbed
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import argparse
import glob
import pickle
from dotenv import load_dotenv
from historical_data_handler import HistoricalDataHandler
from config_loader import get_config

# Load environment variables from .env file
load_dotenv()

# Load configuration
config = get_config()

# Configure logging
logging_config = config.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config.get('level', 'INFO').upper()),
    format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(logging_config.get('file', "divergence_notifier.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - get from config file with environment variable fallbacks
notifications_config = config.get_notifications_config()
CHARTS_DIR = notifications_config.get('charts_directory', os.getenv("CHARTS_DIR", "/Users/isaac/Desktop/Projects/Divergence_Scanner/divergence_charts"))
DISCORD_WEBHOOK_URL = notifications_config.get('discord_webhook_url', os.getenv("DISCORD_WEBHOOK_URL", ""))

class DivergenceNotifier:
    def __init__(self, charts_dir=CHARTS_DIR, webhook_url=DISCORD_WEBHOOK_URL):
        """
        Initialize the Divergence Notifier.
        
        Parameters:
            charts_dir (str): Directory where charts will be saved
            webhook_url (str): Discord webhook URL for notifications
        """
        self.charts_dir = charts_dir
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
        self.historical_data_handler = HistoricalDataHandler()
        
        # Ensure charts directory exists
        os.makedirs(self.charts_dir, exist_ok=True)
        logger.info(f"Charts will be saved to: {self.charts_dir}")
        
        # Track processed results to avoid duplicate notifications
        self.processed_results = set()

    def create_chart(self, symbol, df, divergence_details, divergence_type, indicator_type):
        """
        Create an enhanced chart visualization for the given symbol based on the divergence details.

        Parameters:
            symbol (str): The ticker symbol.
            df (DataFrame): DataFrame with all calculated indicators and swing columns.
            divergence_details (dict): Dictionary with divergence details and trade signals.
            divergence_type (str): Key for the specific divergence (e.g. 'bullish_rsi_strong').
            indicator_type (str): Either 'rsi' or 'ad' to specify which indicator to plot.
            
        Returns:
            filepath (str): Full path to the saved chart image, or None if an error occurred.
        """
        try:
            # Ensure divergence details exist for this type.
            if divergence_type not in divergence_details.get("details", {}):
                self.logger.warning(f"No divergence details found for {divergence_type}")
                return None

            details = divergence_details["details"][divergence_type]
            first_date = details['first_swing']['date']
            second_date = details['second_swing']['date']

            # Use the configured charts directory
            self.logger.info(f"Using charts directory: {self.charts_dir}")

            # Create the figure and subplots.
            fig = plt.figure(figsize=(14, 10))
            gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)
            ax_price = plt.subplot(gs[0])
            ax_indicator = plt.subplot(gs[1], sharex=ax_price)
            ax_volume = plt.subplot(gs[2], sharex=ax_price)
            title = f"{symbol} - {details['type']}"
            fig.suptitle(title, fontsize=16)

            # Convert index to list of dates.
            date_list = df.index.tolist()
            try:
                idx1 = date_list.index(first_date)
                idx2 = date_list.index(second_date)
            except ValueError:
                # If an exact match isn't found, choose the nearest indices.
                idx1 = min(range(len(date_list)), key=lambda i: abs(date_list[i] - first_date))
                idx2 = min(range(len(date_list)), key=lambda i: abs(date_list[i] - second_date))

            # Extract a subset for plotting (30 bars before and after the divergence).
            start_idx = max(0, idx1 - 30)
            end_idx = min(len(df), idx2 + 30)
            plot_df = df.iloc[start_idx:end_idx].copy()

            # Check if a trade signal exists for annotation.
            trade_signal = None
            if "trade_signals" in divergence_details and divergence_type in divergence_details["trade_signals"]:
                trade_signal = divergence_details["trade_signals"][divergence_type]

            # =================== PRICE PANEL ===================
            # Plot price line and add candlesticks.
            ax_price.plot(range(len(plot_df)), plot_df['close'], 'b-', alpha=0.3, linewidth=1)
            for i in range(len(plot_df)):
                if plot_df['close'].iloc[i] >= plot_df['open'].iloc[i]:
                    color = 'green'
                    candle_body_bottom = plot_df['open'].iloc[i]
                    candle_body_top = plot_df['close'].iloc[i]
                else:
                    color = 'red'
                    candle_body_bottom = plot_df['close'].iloc[i]
                    candle_body_top = plot_df['open'].iloc[i]
                ax_price.add_patch(plt.Rectangle(
                    (i - 0.4, candle_body_bottom),
                    0.8,
                    candle_body_top - candle_body_bottom,
                    fill=True,
                    color=color,
                    alpha=0.8
                ))
                # Plot the wicks.
                ax_price.plot([i, i], [plot_df['low'].iloc[i], candle_body_bottom], color='black', linewidth=1)
                ax_price.plot([i, i], [candle_body_top, plot_df['high'].iloc[i]], color='black', linewidth=1)

            # Plot EMAs if available.
            if 'ema_short' in plot_df.columns:
                ax_price.plot(range(len(plot_df)), plot_df['ema_short'], color='blue', linewidth=1, label='EMA 9')
            if 'ema_medium' in plot_df.columns:
                ax_price.plot(range(len(plot_df)), plot_df['ema_medium'], color='purple', linewidth=1, label='EMA 21')
            if 'ema_long' in plot_df.columns:
                ax_price.plot(range(len(plot_df)), plot_df['ema_long'], color='orange', linewidth=1, label='EMA 50')

            # Determine plotting indices for the divergence points.
            idx1_plot = idx1 - start_idx
            idx2_plot = idx2 - start_idx
            if 0 <= idx1_plot < len(plot_df) and 0 <= idx2_plot < len(plot_df):
                price1 = details['first_swing']['price']
                price2 = details['second_swing']['price']
                ax_price.scatter(idx1_plot, price1, color='red', s=100, marker='o', zorder=5)
                ax_price.scatter(idx2_plot, price2, color='red', s=100, marker='o', zorder=5)
                ax_price.plot([idx1_plot, idx2_plot], [price1, price2], 'r--', linewidth=2, zorder=4)
                ax_price.annotate(f"{price1:.2f}", (idx1_plot, price1),
                                    xytext=(10, 10), textcoords='offset points', color='red', fontweight='bold')
                ax_price.annotate(f"{price2:.2f}", (idx2_plot, price2),
                                    xytext=(10, 10), textcoords='offset points', color='red', fontweight='bold')

            # Plot support/resistance levels.
            support_levels = []
            resistance_levels = []
            for i in range(len(plot_df)):
                if 'support' in plot_df.columns and not pd.isna(plot_df['support'].iloc[i]):
                    level = plot_df['support'].iloc[i]
                    if level not in support_levels:
                        support_levels.append(level)
                if 'resistance' in plot_df.columns and not pd.isna(plot_df['resistance'].iloc[i]):
                    level = plot_df['resistance'].iloc[i]
                    if level not in resistance_levels:
                        resistance_levels.append(level)

            # If no support/resistance, derive basic levels from swing lows/highs.
            if not support_levels and not resistance_levels:
                recent_swing_lows = plot_df['swing_low'].dropna()
                recent_swing_highs = plot_df['swing_high'].dropna()
                if not recent_swing_lows.empty:
                    last_price = plot_df['close'].iloc[-1]
                    for level in recent_swing_lows.unique():
                        if level < last_price:
                            support_levels.append(level)
                if not recent_swing_highs.empty:
                    last_price = plot_df['close'].iloc[-1]
                    for level in recent_swing_highs.unique():
                        if level > last_price:
                            resistance_levels.append(level)

            for level in support_levels:
                ax_price.axhline(y=level, color='green', linestyle='-', alpha=0.3, linewidth=1)
                ax_price.annotate(f"Support: {level:.2f}", (0, level),
                                    xytext=(5, 5), textcoords='offset points', color='green', fontsize=8)
            for level in resistance_levels:
                ax_price.axhline(y=level, color='red', linestyle='-', alpha=0.3, linewidth=1)
                ax_price.annotate(f"Resistance: {level:.2f}", (0, level),
                                    xytext=(5, 5), textcoords='offset points', color='red', fontsize=8)

            # If a trade signal is present, mark it.
            if trade_signal:
                signal_time = trade_signal["signal_time"]
                try:
                    signal_idx = date_list.index(signal_time) - start_idx
                except ValueError:
                    signal_idx = idx2_plot
                if 0 <= signal_idx < len(plot_df):
                    entry_price = trade_signal["entry_price"]
                    stop_loss = trade_signal["stop_loss"]
                    take_profit = trade_signal["take_profit"]
                    marker_style = '^' if trade_signal["signal_type"] == "BUY" else 'v'
                    marker_color = 'lime' if trade_signal["signal_type"] == "BUY" else 'red'
                    ax_price.scatter(signal_idx, entry_price, color=marker_color, s=200, marker=marker_style,
                                        zorder=10, edgecolor='black', linewidth=1)
                    ax_price.axhline(y=entry_price, color='blue', linestyle='-', alpha=0.5, linewidth=2)
                    ax_price.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    ax_price.axhline(y=take_profit, color='green', linestyle='--', alpha=0.7, linewidth=2)
                    if trade_signal["signal_type"] == "BUY":
                        ax_price.axhspan(stop_loss, entry_price, color='red', alpha=0.1)
                        ax_price.axhspan(entry_price, take_profit, color='green', alpha=0.1)
                    else:
                        ax_price.axhspan(entry_price, stop_loss, color='red', alpha=0.1)
                        ax_price.axhspan(take_profit, entry_price, color='green', alpha=0.1)
                    ax_price.annotate(f"📊 ENTRY: {entry_price:.2f}", (0, entry_price),
                                        xytext=(5, 0), textcoords='offset points', color='blue', fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
                    ax_price.annotate(f"🛑 STOP: {stop_loss:.2f}", (0, stop_loss),
                                        xytext=(5, 0), textcoords='offset points', color='red', fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
                    ax_price.annotate(f"🎯 TARGET: {take_profit:.2f}", (0, take_profit),
                                        xytext=(5, 0), textcoords='offset points', color='green', fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                    r_r_ratio = reward / risk if risk > 0 else 0
                    ax_price.annotate(
                        f"TRADE SUMMARY\n"
                        f"Type: {trade_signal['signal_type']}\n"
                        f"Entry: {entry_price:.2f}\n"
                        f"Stop: {stop_loss:.2f}\n"
                        f"Target: {take_profit:.2f}\n"
                        f"Risk: {risk:.2f} pts\n"
                        f"Reward: {reward:.2f} pts\n"
                        f"R:R = {r_r_ratio:.2f}",
                        xy=(len(plot_df) - 1, plot_df['high'].max()),
                        xytext=(-120, -20),
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.8),
                        fontsize=10,
                        fontweight='bold'
                    )

            ax_price.set_ylabel('Price')
            ax_price.grid(True, alpha=0.3)
            ax_price.legend(loc='upper left')

            # =================== INDICATOR PANEL ===================
            if indicator_type == 'rsi':
                indicator_name = 'RSI'
                indicator_val1 = details['first_swing']['rsi']
                indicator_val2 = details['second_swing']['rsi']
                ax_indicator.plot(range(len(plot_df)), plot_df['rsi'], color='purple', linewidth=1.5, label=indicator_name)
                ax_indicator.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                ax_indicator.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                ax_indicator.set_ylim(0, 100)
            else:
                indicator_name = 'AD Volume'
                indicator_val1 = details['first_swing']['ad']
                indicator_val2 = details['second_swing']['ad']
                ax_indicator.plot(range(len(plot_df)), plot_df['ad'], color='purple', linewidth=1.5, label=indicator_name)

            if 0 <= idx1_plot < len(plot_df) and 0 <= idx2_plot < len(plot_df):
                ax_indicator.scatter(idx1_plot, indicator_val1, color='red', s=80, marker='o', zorder=5)
                ax_indicator.scatter(idx2_plot, indicator_val2, color='red', s=80, marker='o', zorder=5)
                ax_indicator.plot([idx1_plot, idx2_plot], [indicator_val1, indicator_val2], 'r--', linewidth=2, zorder=4)
                ax_indicator.annotate(f"{indicator_val1:.2f}", (idx1_plot, indicator_val1),
                                        xytext=(5, 5), textcoords='offset points', color='red')
                ax_indicator.annotate(f"{indicator_val2:.2f}", (idx2_plot, indicator_val2),
                                        xytext=(5, 5), textcoords='offset points', color='red')

            ax_indicator.set_ylabel(indicator_name)
            ax_indicator.grid(True, alpha=0.3)
            ax_indicator.legend(loc='upper left')

            # =================== VOLUME PANEL ===================
            for i in range(len(plot_df)):
                color = 'green' if plot_df['close'].iloc[i] >= plot_df['open'].iloc[i] else 'red'
                ax_volume.bar(i, plot_df['volume'].iloc[i], color=color, alpha=0.7, width=0.8)
            volume_ma = plot_df['volume'].rolling(window=20).mean()
            ax_volume.plot(range(len(volume_ma)), volume_ma, color='blue', linewidth=1.5, label='Volume MA (20)')

            if trade_signal:
                try:
                    signal_idx = date_list.index(trade_signal["signal_time"]) - start_idx
                except ValueError:
                    signal_idx = idx2_plot
                if 0 <= signal_idx < len(plot_df):
                    ax_volume.axvline(x=signal_idx, color='blue', linestyle='--', alpha=0.7)
                    ax_volume.annotate("Signal", (signal_idx, plot_df['volume'].iloc[signal_idx]),
                                        xytext=(0, 10), textcoords='offset points', color='blue',
                                        arrowprops=dict(arrowstyle="->", color="blue"))

            ax_volume.set_ylabel('Volume')
            ax_volume.grid(True, alpha=0.3)
            ax_volume.legend(loc='upper left')

            # Format the x-axis with date labels.
            date_ticks = range(0, len(plot_df), max(1, len(plot_df) // 8))
            date_labels = [plot_df.index[i].strftime('%Y-%m-%d %H:%M') for i in date_ticks]
            ax_volume.set_xticks(date_ticks)
            ax_volume.set_xticklabels(date_labels, rotation=45)

            if trade_signal:
                fig.suptitle(f"{symbol} - {details['type']} - {trade_signal['signal_type']} Signal", fontsize=16)
                confidence = trade_signal.get('confidence', 'medium').upper()
                confidence_color = 'green' if confidence == 'HIGH' else 'orange' if confidence == 'MEDIUM' else 'red'
                fig.text(0.5, 0.01, f"Signal Confidence: {confidence}", ha='center', va='bottom', fontsize=12,
                            color=confidence_color, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=confidence_color, alpha=0.8))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_symbol = symbol.replace('/', '_')
            direction = "bullish" if "bullish" in divergence_type else "bearish"
            strength = divergence_type.split('_')[-1]
            filename = f"{safe_symbol}_{direction}_{indicator_type}_{strength}_{timestamp}.png"
            filepath = os.path.join(self.charts_dir, filename)
            self.logger.info(f"Attempting to save chart to: {filepath}")
            plt.savefig(filepath, dpi=120)
            plt.close(fig)
            self.logger.info(f"Created chart at {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error creating chart: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def process_results(self, results_dict, df_dict):
        """
        Process a dictionary of divergence results and dataframes.
        
        Parameters:
            results_dict (dict): Dictionary of divergence results keyed by symbol
            df_dict (dict): Dictionary of dataframes keyed by symbol
        """
        for symbol, result_data in results_dict.items():
            if symbol not in df_dict:
                self.logger.warning(f"No dataframe available for {symbol}, skipping")
                continue
                
            # Create a unique key to track if we've already processed this result
            timestamp = result_data.get('timestamp', '')
            result_key = f"{symbol}_{timestamp}"
            
            if result_key in self.processed_results:
                self.logger.info(f"Already processed {result_key}, skipping")
                continue
                
            # Check if there are any divergences
            divergence_results = result_data.get('data', {})
            has_divergence = False
            
            # Check RSI divergences
            rsi_bull = divergence_results.get("rsi_divergences", {}).get("bullish", {})
            rsi_bear = divergence_results.get("rsi_divergences", {}).get("bearish", {})
            if any(rsi_bull.values()) or any(rsi_bear.values()):
                has_divergence = True
                
            # Check AD volume divergences
            adv_bull = divergence_results.get("adv_divergences", {}).get("bullish", {})
            adv_bear = divergence_results.get("adv_divergences", {}).get("bearish", {})
            if any(adv_bull.values()) or any(adv_bear.values()):
                has_divergence = True
                
            if has_divergence:
                self.logger.info(f"Processing divergences for {symbol}")
                self.send_discord_notification(symbol, divergence_results, df_dict[symbol])
                self.processed_results.add(result_key)
            else:
                self.logger.info(f"No divergences found for {symbol}")

    def send_discord_notification(self, symbol, divergence_results, df_with_swings):
        """
        Send enhanced divergence notifications to Discord with trading signals and charts.

        Parameters:
            symbol (str): Ticker symbol (e.g. '/ES').
            divergence_results (dict): Dictionary containing divergence flags, details, and trade signals.
            df_with_swings (DataFrame): DataFrame containing calculated indicators and swing columns.
        """
        try:
            # Get the current time string.
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the main webhook embed.
            webhook = DiscordWebhook(url=self.webhook_url)
            main_embed = DiscordEmbed(
                title=f"🔍 Divergence Alert - {symbol}",
                color=242424,
                description=f"Detected at {current_time}\nCurrent Price: {df_with_swings['close'].iloc[-1]:.2f}"
            )
            
            # Add market context if available.
            if 'trend' in df_with_swings.columns:
                trend = df_with_swings['trend'].iloc[-1]
                main_embed.add_embed_field(
                    name="Market Trend",
                    value=f"{'🟢' if trend=='bullish' else '🔴' if trend=='bearish' else '⚪'} {trend.capitalize()}",
                    inline=True
                )
            if 'market_session' in df_with_swings.columns:
                session = df_with_swings['market_session'].iloc[-1]
                main_embed.add_embed_field(name="Market Session", value=session, inline=True)
            if 'adx' in df_with_swings.columns:
                adx_value = df_with_swings['adx'].iloc[-1]
                trend_strength = "Strong" if adx_value > 25 else "Moderate" if adx_value > 15 else "Weak"
                main_embed.add_embed_field(
                    name="Trend Strength",
                    value=f"{trend_strength} (ADX: {adx_value:.1f})",
                    inline=True
                )

            active_signals = []
            chart_files = []
            divergence_descriptions = []

            # Process RSI Bullish Divergences.
            rsi_bull = divergence_results.get("rsi_divergences", {}).get("bullish", {})
            for div_type in ["strong", "medium", "weak", "hidden"]:
                div_key = f"bullish_rsi_{div_type}"
                if rsi_bull.get(div_type, False) and (div_key in divergence_results.get("details", {})):
                    details = divergence_results["details"][div_key]
                    divergence_descriptions.append(
                        f"🟢 {div_type.capitalize()} Bullish RSI Divergence\n"
                        f"First: {details['first_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['first_swing']['price']:.2f}, RSI: {details['first_swing']['rsi']:.2f}\n"
                        f"Second: {details['second_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['second_swing']['price']:.2f}, RSI: {details['second_swing']['rsi']:.2f}"
                    )
                    chart_path = self.create_chart(symbol, df_with_swings, divergence_results, div_key, 'rsi')
                    if chart_path:
                        chart_files.append(chart_path)

            # Process RSI Bearish Divergences.
            rsi_bear = divergence_results.get("rsi_divergences", {}).get("bearish", {})
            for div_type in ["strong", "medium", "weak", "hidden"]:
                div_key = f"bearish_rsi_{div_type}"
                if rsi_bear.get(div_type, False) and (div_key in divergence_results.get("details", {})):
                    details = divergence_results["details"][div_key]
                    divergence_descriptions.append(
                        f"🔴 {div_type.capitalize()} Bearish RSI Divergence\n"
                        f"First: {details['first_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['first_swing']['price']:.2f}, RSI: {details['first_swing']['rsi']:.2f}\n"
                        f"Second: {details['second_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['second_swing']['price']:.2f}, RSI: {details['second_swing']['rsi']:.2f}"
                    )
                    chart_path = self.create_chart(symbol, df_with_swings, divergence_results, div_key, 'rsi')
                    if chart_path:
                        chart_files.append(chart_path)

            # Process AD Volume Bullish Divergences.
            adv_bull = divergence_results.get("adv_divergences", {}).get("bullish", {})
            for div_type in ["strong", "medium", "weak", "hidden"]:
                div_key = f"bullish_adv_{div_type}"
                if adv_bull.get(div_type, False) and (div_key in divergence_results.get("details", {})):
                    details = divergence_results["details"][div_key]
                    divergence_descriptions.append(
                        f"🟢 {div_type.capitalize()} Bullish AD Volume Divergence\n"
                        f"First: {details['first_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['first_swing']['price']:.2f}, AD: {details['first_swing']['ad']:.2f}\n"
                        f"Second: {details['second_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['second_swing']['price']:.2f}, AD: {details['second_swing']['ad']:.2f}"
                    )
                    chart_path = self.create_chart(symbol, df_with_swings, divergence_results, div_key, 'ad')
                    if chart_path:
                        chart_files.append(chart_path)

            # Process AD Volume Bearish Divergences.
            adv_bear = divergence_results.get("adv_divergences", {}).get("bearish", {})
            for div_type in ["strong", "medium", "weak", "hidden"]:
                div_key = f"bearish_adv_{div_type}"
                if adv_bear.get(div_type, False) and (div_key in divergence_results.get("details", {})):
                    details = divergence_results["details"][div_key]
                    divergence_descriptions.append(
                        f"🔴 {div_type.capitalize()} Bearish AD Volume Divergence\n"
                        f"First: {details['first_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['first_swing']['price']:.2f}, AD: {details['first_swing']['ad']:.2f}\n"
                        f"Second: {details['second_swing']['date'].strftime('%Y-%m-%d %H:%M')}, Price: {details['second_swing']['price']:.2f}, AD: {details['second_swing']['ad']:.2f}"
                    )
                    chart_path = self.create_chart(symbol, df_with_swings, divergence_results, div_key, 'ad')
                    if chart_path:
                        chart_files.append(chart_path)

            # Append the divergence descriptions to the main embed.
            if divergence_descriptions:
                main_embed.description += "\n\n**Detected Divergences:**"
                for i, desc in enumerate(divergence_descriptions):
                    main_embed.add_embed_field(name=f"Divergence #{i+1}", value=desc, inline=False)

            # Include a summary of active trade signals if present.
            if "trade_signals" in divergence_results and divergence_results["trade_signals"]:
                signals_summary = "\n".join([
                    f"{'🟢' if signal['signal_type'] == 'BUY' else '🔴'} {signal['signal_type']} at {signal['entry_price']:.2f} "
                    f"(Stop: {signal['stop_loss']:.2f}, Target: {signal['take_profit']:.2f}, R:R: {signal['reward_risk_ratio']:.2f})"
                    for signal in divergence_results["trade_signals"].values()
                ])
                main_embed.add_embed_field(name="📊 Active Trade Signals", value=signals_summary, inline=False)

            # Add technical summary information.
            main_embed.add_embed_field(
                name="📈 Technical Summary",
                value=(
                    f"RSI: {df_with_swings['rsi'].iloc[-1]:.1f} "
                    f"({'Overbought' if df_with_swings['rsi'].iloc[-1] > 70 else 'Oversold' if df_with_swings['rsi'].iloc[-1] < 30 else 'Neutral'})\n"
                    f"MACD: {df_with_swings['macd'].iloc[-1]:.2f} "
                    f"({'Bullish' if df_with_swings['macd'].iloc[-1] > df_with_swings['macd_signal'].iloc[-1] else 'Bearish'})"
                ),
                inline=False
            )

            main_embed.set_footer(text="Timeframe: 15-minute | Scan will repeat every 60 seconds. You will receive alerts for new divergences only.")
            webhook.add_embed(main_embed)

            # Attach any generated chart files.
            for chart_path in chart_files:
                with open(chart_path, "rb") as f:
                    webhook.add_file(file=f.read(), filename=os.path.basename(chart_path))

            response = webhook.execute()
            if response.status_code in (200, 204):
                self.logger.info(f"Discord notification sent for {symbol} with {len(chart_files)} charts")
            else:
                self.logger.error(f"Failed to send Discord notification: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error sending Discord notification: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")


def main():
    """Main function to run the divergence notifier."""
    parser = argparse.ArgumentParser(description="Divergence Notifier Script")
    parser.add_argument("--webhook", help="Discord webhook URL", default=DISCORD_WEBHOOK_URL)
    parser.add_argument("--charts-dir", help="Directory to save charts", default=CHARTS_DIR)
    parser.add_argument("--results-file", help="Path to pickle file containing scanner results", required=False)
    parser.add_argument("--live-monitor", help="Monitor the scanner's results dictionary in realtime", action="store_true")
    parser.add_argument("--poll-interval", help="Polling interval in seconds for live monitoring", type=int, default=60)
    args = parser.parse_args()
    
    notifier = DivergenceNotifier(charts_dir=args.charts_dir, webhook_url=args.webhook)
    
    if args.live_monitor:
        # Setup for live monitoring
        from Divergence_calculator import DivergenceScanner
        scanner = DivergenceScanner()
        
        try:
            logger.info("Starting live monitoring of divergence scanner results")
            last_check_time = 0
            
            while True:
                current_time = time.time()
                if current_time - last_check_time >= args.poll_interval:
                    # Process any new results
                    results_dict = scanner.results
                    
                    if results_dict:
                        logger.info(f"Found results for {len(results_dict)} symbols")
                        
                        # Get dataframes with indicators and swings for any symbols with results
                        df_dict = {}
                        for symbol in results_dict.keys():
                            try:
                                df = scanner.get_historical_data(symbol, period_type="day", period=10, frequency_type="minute", frequency=15)
                                if df is not None and not df.empty:
                                    df = scanner.calculate_indicators(df)
                                    df = scanner.detect_price_swings(df)
                                    df_dict[symbol] = df
                                    logger.info(f"Successfully prepared data for {symbol}")
                            except Exception as e:
                                logger.error(f"Error getting data for {symbol}: {str(e)}")
                        
                        # Process results
                        logger.info(f"Processing results for {len(df_dict)} symbols with complete data")
                        notifier.process_results(results_dict, df_dict)
                    else:
                        logger.info("No results found in scanner yet")
                    
                    last_check_time = current_time
                    logger.info(f"Next check in {args.poll_interval} seconds")
                
                # Small sleep to prevent cpu hogging
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    elif args.results_file:
        # Process results from pickle file
        try:
            with open(args.results_file, 'rb') as f:
                results_data = pickle.load(f)
                
            if isinstance(results_data, tuple) and len(results_data) >= 2:
                # Unpack tuple of (results_dict, df_dict)
                results_dict, df_dict = results_data
                notifier.process_results(results_dict, df_dict)
            elif isinstance(results_data, dict):
                # Single results dict
                results_dict = results_data
                
                # Get dataframes with indicators and swings for any symbols with results
                df_dict = {}
                historical_data_handler = HistoricalDataHandler()
                
                for symbol in results_dict.keys():
                    try:
                        # Create a DivergenceScanner instance just to use its methods
                        from Divergence_calculator import DivergenceScanner
                        temp_scanner = DivergenceScanner()
                        
                        df = temp_scanner.get_historical_data(symbol, period_type="day", period=10, frequency_type="minute", frequency=15)
                        if df is not None and not df.empty:
                            df = temp_scanner.calculate_indicators(df)
                            df = temp_scanner.detect_price_swings(df)
                            df_dict[symbol] = df
                    except Exception as e:
                        logger.error(f"Error getting data for {symbol}: {str(e)}")
                
                # Process results
                notifier.process_results(results_dict, df_dict)
            else:
                logger.error(f"Unrecognized results format in {args.results_file}")
                
        except Exception as e:
            logger.error(f"Error processing results file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        # No specific mode selected
        parser.print_help()
        logger.info("Please specify either --live-monitor or --results-file")


if __name__ == "__main__":
    main()
