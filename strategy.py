#!/usr/bin/env python3
name = "6h_WeeklyTrend_Donchian_Breakout_Volume"
timeframe = "6h"
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
    
    # Load daily data ONCE for weekly trend and pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (from daily data, but we'll use 1w data for proper weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian channels for breakout levels
    donch_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Daily volume average for confirmation
    vol_avg_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 6h ATR for stop management
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_avg_20_aligned[i] * 1.5
        
        if position == 0:
            # Long: breakout above Donchian high in weekly uptrend with volume
            if close[i] > donch_high_20_aligned[i] and vol_condition and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in weekly downtrend with volume
            elif close[i] < donch_low_20_aligned[i] and vol_condition and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or weekly trend fails
            midpoint = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2
            if close[i] < midpoint or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or weekly trend fails
            midpoint = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2
            if close[i] > midpoint or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Donchian breakout with weekly trend filter and volume confirmation
# - Uses daily Donchian channels (20-period) for breakout levels
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x daily average) reduces false breakouts
# - Exits when price returns to Donchian midpoint or weekly trend changes
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Position size 0.25 targets ~15-35 trades/year to stay within limits
# - Novel combination: Daily Donchian + weekly trend + volume on 6h timeframe
# - Aims for 60-140 total trades over 4 years (15-35/year) to stay within limits
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Donchian breakouts provide clear structure with defined support/resistance levels
# - Volume confirmation adds validity to breakouts
# - Weekly trend ensures we only trade in the direction of higher timeframe momentum