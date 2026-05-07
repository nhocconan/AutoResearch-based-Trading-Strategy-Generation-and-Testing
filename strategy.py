#!/usr/bin/env python3
name = "1d_Donchian20_WeeklyTrend_Volume"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 10-period average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 10)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_10[i] * 2.0
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > high_max_20[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif close[i] < low_min_20[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or volume drops
            if close[i] < low_min_20[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to Donchian high or volume drops
            if close[i] > high_max_20[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly trend and volume confirmation
# - Donchian breakouts capture significant price moves in both bull and bear markets
# - Weekly EMA(34) trend filter ensures trades align with higher timeframe momentum
# - Volume spike (2x average) confirms institutional participation and reduces false breakouts
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Weekly trend filter reduces whipsaws vs using same timeframe
# - Volume confirmation improves signal quality
# - Designed for BTC/ETH primary focus with SOL as secondary
# - Aims for 60-120 total trades over 4 years (15-30/year) to stay within limits