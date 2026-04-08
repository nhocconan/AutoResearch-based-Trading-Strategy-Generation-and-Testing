#!/usr/bin/env python3
# 4h_1w_rsi_extreme_v1
# Hypothesis: Use weekly RSI extremes for directional bias and daily RSI for entry timing.
# In weekly uptrend (RSI>50): go long when daily RSI crosses above 50 with volume confirmation.
# In weekly downtrend (RSI<50): go short when daily RSI crosses below 50 with volume confirmation.
# Exit when daily RSI crosses back to 50 or weekly trend flips.
# Uses weekly RSI(14) for trend filter and daily RSI(14) for entry signals.
# Volume filter ensures momentum confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_rsi_extreme_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly RSI for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily RSI for entry signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: daily RSI crosses below 50 or weekly RSI flips to bearish (<50)
            if rsi_1d_aligned[i] < 50 or rsi_1w_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: daily RSI crosses above 50 or weekly RSI flips to bullish (>50)
            if rsi_1d_aligned[i] > 50 or rsi_1w_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly bullish (RSI>50) and daily RSI crosses above 50 with volume surge
            if (rsi_1w_aligned[i] > 50 and rsi_1d_aligned[i] > 50 and 
                rsi_1d_aligned[i-1] <= 50 and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly bearish (RSI<50) and daily RSI crosses below 50 with volume surge
            elif (rsi_1w_aligned[i] < 50 and rsi_1d_aligned[i] < 50 and 
                  rsi_1d_aligned[i-1] >= 50 and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals