#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Trend_Volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channel (20-day) on daily data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 5-day average
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channel and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_5[i] * 2.0
            weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            
            if close[i] > high_max[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif close[i] < low_min[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or volume drops
            if close[i] < low_min[i] or volume[i] < vol_ma_5[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to Donchian high or volume drops
            if close[i] > high_max[i] or volume[i] < vol_ma_5[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly trend and volume confirmation
# - Donchian breakout captures momentum in both bull and bear markets
# - Weekly EMA(20) filter ensures we trade with the higher timeframe trend
# - Volume spike (2x average) confirms institutional participation
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~15-30 trades/year to avoid fee drag
# - Uses actual weekly data for trend filter, not resampled
# - Designed for low trade frequency and high win rate in volatile markets