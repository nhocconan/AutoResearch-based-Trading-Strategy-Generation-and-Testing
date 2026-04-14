#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Camarilla pivot levels (S1, R1) with volume confirmation
# - Long when price crosses above S1 with volume > 1.5x 48-period average
# - Short when price crosses below R1 with volume > 1.5x 48-period average
# - Uses 1-day EMA200 as trend filter: only long when price > EMA200, short when price < EMA200
# - Exits on opposite pivot level touch (R1 for longs, S1 for shorts)
# - Position size 0.25 to balance risk and returns
# - Target: 50-150 trades over 4 years (12-37/year) to avoid excessive fees

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: S1 = close - (high - low) * 1.1/6, R1 = close + (high - low) * 1.1/6
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 6
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 6
    
    # Calculate 1-day EMA200 for trend filter
    close_series = pd.Series(close_1d)
    ema200_1d = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume filter: 48-period average (4 days of 12h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=48, min_periods=48).mean().values
    
    # Align 1d data to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            continue
        
        if position == 0:
            # Long: Price crosses above S1 with volume and trend filter
            if (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and
                volume[i] > vol_ma[i] * 1.5 and
                close[i] > ema200_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below R1 with volume and trend filter
            elif (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and
                  volume[i] > vol_ma[i] * 1.5 and
                  close[i] < ema200_aligned[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price touches R1
            if close[i] >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price touches S1
            if close[i] <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0