#!/usr/bin/env python3
name = "6h_1w_1d_Camarilla_WeeklyTrend_DailyPivot_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly trend filter: EMA(50) on weekly close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily pivot levels from previous day
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (S3/R3 for reversal, S4/R4 for breakout)
    s3 = prev_close_1d - (range_1d * 1.1 / 4)
    r3 = prev_close_1d + (range_1d * 1.1 / 4)
    s4 = prev_close_1d - (range_1d * 1.1 / 2)
    r4 = prev_close_1d + (range_1d * 1.1 / 2)
    
    # Align daily levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume spike: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above R3 with weekly uptrend and volume
            # OR price breaks above S4 with weekly uptrend (strong breakout)
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            long_signal = (close[i] > r3_aligned[i] or close[i] > s4_aligned[i]) and vol_condition and weekly_uptrend
            
            # Short: price below S3 with weekly downtrend and volume
            # OR price breaks below R4 with weekly downtrend (strong breakdown)
            short_signal = (close[i] < s3_aligned[i] or close[i] < r4_aligned[i]) and vol_condition and not weekly_uptrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to R3 or weekly trend turns down
            if close[i] < r3_aligned[i] or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to S3 or weekly trend turns up
            if close[i] > s3_aligned[i] or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla S3/R3/S4/R4 breakout with weekly trend filter
# - Weekly EMA(50) determines primary trend (bull/bear regime)
# - In weekly uptrend: long when price breaks above R3 (reversal) or S4 (breakout) with volume
# - In weekly downtrend: short when price breaks below S3 (reversal) or R4 (breakdown) with volume
# - Uses S3/R3 for mean-reversion entries in trend, S4/R4 for momentum breakouts
# - Volume spike (2x average) confirms institutional participation
# - Exits when price returns to S3/R3 or weekly trend changes
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Works in both bull (buy breaks in uptrend) and bear (sell breaks in downtrend)
# - Weekly trend filter avoids counter-trend trades in strong regimes
# - Combines reversal and breakout logic for adaptability to market conditions