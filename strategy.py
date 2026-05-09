#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly RSI mean reversion and volume confirmation.
# In high RSI (>70) conditions, price tends to revert to mean; in low RSI (<30), price tends to bounce.
# Uses weekly RSI for regime and daily price action for entry timing with volume filter.
# Target: 20-80 total trades over 4 years (5-20/year) with size 0.25.

name = "1d_WeeklyRSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Volume filter: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if RSI not ready
        if np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly RSI < 30 (oversold) + volume filter
            if rsi_1w_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly RSI > 70 (overbought) + volume filter
            elif rsi_1w_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly RSI > 50 (mean reversion complete)
            if rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly RSI < 50 (mean reversion complete)
            if rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals