#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_Volume_Trend
# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and EMA trend filter
# Camarilla levels from prior day provide statistically significant support/resistance
# EMA34 filter ensures trades align with intermediate trend, reducing false breakouts
# Volume > 1.5x 20-period average confirms institutional participation
# Target: 50-150 trades over 4 years (12-37/year) with controlled risk via position sizing

name = "12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
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
    
    # EMA34 for trend filter - calculated on 12h data
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2.0 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(ema[i-1]):
                ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
            else:
                ema[i] = np.nan
        return ema
    
    # 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    ema34_12h = calculate_ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla calculations
    rang = ph - pl
    r1 = pc + (rang * 1.1 / 12)
    s1 = pc - (rang * 1.1 / 12)
    r4 = pc + (rang * 1.1 / 2)
    s4 = pc - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and price above EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and price below EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend reverses (price below EMA34)
            if (close[i] < s1_aligned[i]) or (close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend reverses (price above EMA34)
            if (close[i] > r1_aligned[i]) or (close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals