#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA20_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load 1d data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Pivot Points from previous 1d
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align Pivot levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and Pivots
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with 12h uptrend and volume
            if (close[i] > r1_aligned[i] and close[i] > ema_12h_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with 12h downtrend and volume
            elif (close[i] < s1_aligned[i] and close[i] < ema_12h_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend change
            if close[i] < s1_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend change
            if close[i] > r1_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA(20) trend filter and volume confirmation.
# Pivot levels from prior day provide key support/resistance. Breaking R1/S1 indicates momentum.
# 12h EMA filter ensures alignment with higher timeframe trend, reducing whipsaw in choppy markets.
# Volume confirms institutional participation. Position size 0.25 limits drawdown. Target: ~25-35 trades/year.