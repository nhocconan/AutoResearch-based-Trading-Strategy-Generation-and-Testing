#!/usr/bin/env python3
name = "4h_Adaptive_Pivot_Trend_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for pivot levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h pivot levels (using previous period's high/low/close)
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    r1_12h = 2 * pivot_12h - prev_low_12h
    s1_12h = 2 * pivot_12h - prev_high_12h
    r2_12h = pivot_12h + (prev_high_12h - prev_low_12h)
    s2_12h = pivot_12h - (prev_high_12h - prev_low_12h)
    
    # Align pivot levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # 12h EMA for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 4h volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_10[i] * 1.5
        
        if position == 0:
            # Long: price breaks above S1 with volume in 12h uptrend
            if close[i] > s1_12h_aligned[i] and vol_condition and ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 with volume in 12h downtrend
            elif close[i] < r1_12h_aligned[i] and vol_condition and ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back below pivot or trend changes
            if close[i] < pivot_12h_aligned[i] or ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above pivot or trend changes
            if close[i] > pivot_12h_aligned[i] or ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla pivot breakouts with 12h trend and volume filter
# - Uses 12h pivot levels (R1, S1, pivot) from previous period for structure
# - Long when price breaks above S1 with volume spike in 12h uptrend
# - Short when price breaks below R1 with volume spike in 12h downtrend
# - Exit when price returns to pivot level or 12h trend changes
# - Volume confirmation (1.5x average) reduces false breakouts
# - Position size 0.25 balances return and risk
# - Designed for 15-35 trades/year to avoid fee drag
# - Works in bull markets (buying S1 breaks in uptrend) and bear markets (selling R1 breaks in downtrend)
# - Pivot levels provide objective support/resistance based on prior 12h action
# - 12h trend filter ensures alignment with higher timeframe momentum
# - Novel: Combines pivot breakouts with volume and trend on 4h/12h timeframe not recently tried
# - Aims for 60-140 total trades over 4 years to stay within limits