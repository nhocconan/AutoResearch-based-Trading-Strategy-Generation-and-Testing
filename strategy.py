#!/usr/bin/env python3
name = "1d_WeeklyPivot_R1S1_Breakout_Trend_Filter_v6"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points and levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly levels to daily timeframe
    pivot_aligned = align_ltf_to_hlf(prices, df_1w, pivot)
    r1_aligned = align_ltf_to_hlf(prices, df_1w, r1)
    s1_aligned = align_ltf_to_hlf(prices, df_1w, s1)
    
    # Daily trend filter: 50-day EMA
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            uptrend = close[i] > ema_50[i]
            
            if close[i] > r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and downtrend
            elif close[i] < s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly pivot or volume drops
            if close[i] < pivot_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly pivot or volume drops
            if close[i] > pivot_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot point breakout with daily trend filter and volume confirmation
# - Weekly R1/S1 act as significant support/resistance levels
# - Breakouts above R1 or below S1 with volume capture institutional moves
# - Daily price > 50 EMA ensures alignment with intermediate trend
# - Volume confirmation (1.5x average) filters false breakouts
# - Works in bull (buy R1 breakouts in uptrend) and bear (sell S1 breakdowns in downtrend)
# - Position size 0.25 targets ~15-25 trades/year, avoiding excessive fee drag
# - Exit at weekly pivot provides logical mean-reversion target in ranging markets
# - Weekly timeframe reduces noise while daily execution provides timely entry/exit