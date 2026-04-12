#!/usr/bin/env python3
"""
1d_1w_RSI_Trend_Filter
Hypothesis: On daily timeframe, use weekly RSI trend filter with daily RSI mean reversion.
Enter long when weekly RSI > 50 (bullish trend) and daily RSI < 30 (oversold).
Enter short when weekly RSI < 50 (bearish trend) and daily RSI > 70 (overbought).
Exit when daily RSI crosses back to 50.
Uses 0.25 position sizing to balance risk and return.
Designed to capture mean reversion within the prevailing weekly trend, effective in both bull and bear markets.
Target: 20-60 total trades over 4 years (5-15/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === WEEKLY RSI TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.zeros_like(gain_1w)
    avg_loss_1w = np.zeros_like(loss_1w)
    avg_gain_1w[13] = np.mean(gain_1w[1:14])
    avg_loss_1w[13] = np.mean(loss_1w[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
        avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.full_like(avg_gain_1w, 50.0), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_1d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === DAILY RSI MEAN REVERSION SIGNAL ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if np.isnan(rsi_1w_1d[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: weekly uptrend (RSI > 50) + daily oversold (RSI < 30)
        long_signal = (rsi_1w_1d[i] > 50) and (rsi[i] < 30)
        
        # Short: weekly downtrend (RSI < 50) + daily overbought (RSI > 70)
        short_signal = (rsi_1w_1d[i] < 50) and (rsi[i] > 70)
        
        # Exit: daily RSI crosses back to 50 (mean reversion complete)
        exit_long = (position == 1) and (rsi[i] >= 50)
        exit_short = (position == -1) and (rsi[i] <= 50)
        
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