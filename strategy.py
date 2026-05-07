#!/usr/bin/env python3
"""
4H_RSI_Trend_Filter_1D_Volume_Breakout_v1
Hypothesis: Combine 4h RSI trend filter with 1d volume breakout to capture momentum in both bull and bear markets.
- Long when 4h RSI > 55 and price breaks above 1d high with volume > 2x average
- Short when 4h RSI < 45 and price breaks below 1d low with volume > 2x average
- Exit when RSI returns to neutral zone (45-55)
- Volume confirmation reduces false breakouts; RSI filter ensures trend alignment.
- Designed for moderate trade frequency (~25-40/year) to avoid fee drag.
"""
name = "4H_RSI_Trend_Filter_1D_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = pd.Series(df_4h['close'])
    delta = close_4h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Get 1d data for high/low levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Align 1d high/low to 4t
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume filter: current volume > 2x 20-day average volume
    volume_filter = volume > (vol_avg_1d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 55 and price breaks above 1d high with volume confirmation
            if (rsi_aligned[i] > 55 and 
                high[i] > high_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 and price breaks below 1d low with volume confirmation
            elif (rsi_aligned[i] < 45 and 
                  low[i] < low_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI returns to neutral zone (45-55)
            if position == 1 and rsi_aligned[i] < 55:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_aligned[i] > 45:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals