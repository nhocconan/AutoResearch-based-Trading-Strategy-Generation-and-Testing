#!/usr/bin/env python3
"""
1d_1w_Weekly_RSI_Momentum
Hypothesis: Use weekly RSI to identify momentum extremes on daily timeframe.
Long when weekly RSI > 50 and daily RSI crosses above 30 from below (bullish momentum in uptrend).
Short when weekly RSI < 50 and daily RSI crosses below 70 from above (bearish momentum in downtrend).
Weekly RSI acts as trend filter, daily RSI provides entry timing.
Designed to capture medium-term moves in both bull and bear markets with low trade frequency.
Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_RSI_Momentum"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly RSI(14)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False).mean().values
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === DAILY RSI FOR ENTRY TIMING ===
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if not ready
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly RSI trend filter
        weekly_bullish = rsi_1w_aligned[i] > 50
        weekly_bearish = rsi_1w_aligned[i] < 50
        
        # Daily RSI signals
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        
        # Long: weekly bullish + daily RSI crosses above 30 from below
        long_signal = weekly_bullish and (rsi_prev <= 30) and (rsi_now > 30)
        
        # Short: weekly bearish + daily RSI crosses below 70 from above
        short_signal = weekly_bearish and (rsi_prev >= 70) and (rsi_now < 70)
        
        # Exit: opposite weekly RSI extreme
        exit_long = position == 1 and rsi_1w_aligned[i] < 40
        exit_short = position == -1 and rsi_1w_aligned[i] > 60
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals