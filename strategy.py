#!/usr/bin/env python3
"""
1d_1W_Camarilla_S1_R1_Breakout_Volume_Sparse_V1
Hypothesis: Weekly trend filter + daily Camarilla S1/R1 breakout with volume confirmation.
Long when price > weekly EMA20 (uptrend) and breaks above daily R1 with volume > 1.5x average.
Short when price < weekly EMA20 (downtrend) and breaks below daily S1 with volume > 1.5x average.
Position size: 0.25. Target: 8-20 trades/year (32-80 total over 4 years) to avoid fee drag.
Works in bull via long bias, in bear via short bias, both require trend + breakout + volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align daily data to 1d timeframe (no additional alignment needed as prices is already 1d)
    r1_aligned = r1  # already 1d
    s1_aligned = s1  # already 1d
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: uptrend + break above R1 + volume
            if close[i] > ema_20_1w_aligned[i] and close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + break below S1 + volume
            elif close[i] < ema_20_1w_aligned[i] and close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or price returns below R1
            if close[i] < ema_20_1w_aligned[i] or close[i] < r1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or price returns above S1
            if close[i] > ema_20_1w_aligned[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_S1_R1_Breakout_Volume_Sparse_V1"
timeframe = "1d"
leverage = 1.0