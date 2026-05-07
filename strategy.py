#!/usr/bin/env python3
name = "1d_1w_Camarilla_Pullback_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: EMA(10) on weekly close
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily price action: previous day's high/low/close for Camarilla
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Calculate daily Camarilla levels from previous day
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Volume filter: 5-day average volume
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 5)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(s1[i]) or 
            np.isnan(r1[i]) or np.isnan(vol_ma_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to S1 in weekly uptrend with volume
            vol_condition = volume[i] > vol_ma_5[i] * 1.5
            weekly_uptrend = ema_10_1w_aligned[i] > ema_10_1w_aligned[i-1]
            
            if low[i] <= s1[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: pullback to R1 in weekly downtrend with volume
            elif high[i] >= r1[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves above pivot or volume drops
            if high[i] > pivot[i] or volume[i] < vol_ma_5[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves below pivot or volume drops
            if low[i] < pivot[i] or volume[i] < vol_ma_5[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla S1/R1 pullback strategy with weekly trend filter
# - In weekly uptrend: buy pullbacks to daily S1 (support) with volume confirmation
# - In weekly downtrend: sell pullbacks to daily R1 (resistance) with volume confirmation
# - Uses actual daily price action (not intraday) for cleaner signals
# - Weekly EMA(10) filter ensures trading with the higher timeframe trend
# - Volume spike (1.5x average) confirms institutional interest at pullback
# - Exit when price returns to daily pivot or volume wanes
# - Position size 0.25 targets ~15-25 trades/year, minimizing fee drag
# - Works in both bull (buy S1 pullbacks in uptrend) and bear (sell R1 pullbacks in downtrend) markets
# - Designed for 1d timeframe to reduce trade frequency and improve generalization