#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_4hTrend_1dVol
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
# Enters long when price breaks above 4h R1, close > 1d EMA50 (uptrend), and volume > 1.5x average.
# Enters short when price breaks below 4h S1, close < 1d EMA50 (downtrend), and volume > 1.5x average.
# Exits when price returns to opposite S1/R1 level. Uses session filter (08-20 UTC) to reduce noise.
# Target: 15-30 trades/year by combining 4h structure with 1d trend and volume filters.

name = "1h_Camarilla_R1_S1_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h period's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + 1.1 * hl_range_4h / 12
    s1_4h = close_4h - 1.1 * hl_range_4h / 12
    
    # Align 4h levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 for trend filter (1d)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: 1.5x average volume (24-period for 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h R1, close > 1d EMA50 (uptrend), volume spike (>1.5x)
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h S1, close < 1d EMA50 (downtrend), volume spike (>1.5x)
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to or below 4h S1 (opposite level)
            if close[i] <= s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to or above 4h R1 (opposite level)
            if close[i] >= r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals