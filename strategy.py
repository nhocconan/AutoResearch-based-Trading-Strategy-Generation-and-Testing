#!/usr/bin/env python3
"""
1d_1w_keltner_reversion_v1
Hypothesis: Daily mean reversion at Keltner Channel extremes with weekly trend filter.
Buy when price touches lower KC(20,2) in weekly uptrend, sell when touches upper KC in weekly downtrend.
Uses volume confirmation to avoid false signals. Designed for low trade frequency (10-30/year).
Works in bull/bear by aligning with weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Keltner
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend and KC center
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Weekly ATR for KC width
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1w = pd.Series(tr1).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly Keltner Channels
    upper_1w = ema20_1w + 2 * atr_1w
    lower_1w = ema20_1w - 2 * atr_1w
    
    # Align to daily
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches lower KC in weekly uptrend with volume
        if (low[i] <= lower_1w_aligned[i] and close[i] > ema20_1w_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price touches upper KC in weekly downtrend with volume
        elif (high[i] >= upper_1w_aligned[i] and close[i] < ema20_1w_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to weekly EMA20 or opposite KC touch
        elif position == 1 and high[i] >= upper_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and low[i] <= lower_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_keltner_reversion_v1"
timeframe = "1d"
leverage = 1.0