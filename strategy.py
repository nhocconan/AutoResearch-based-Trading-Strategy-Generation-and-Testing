#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Trend_Volume_v1"
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
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume spike detection: 4-period average (4 days)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 4)  # Wait for weekly EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Donchian with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > high_20[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Donchian with volume and weekly downtrend
            elif close[i] < low_20[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below lower Donchian or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above upper Donchian or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(34) trend and volume confirmation
# - Daily Donchian breakout provides clear entry/exit levels based on price action
# - Weekly EMA(34) trend filter ensures trades align with higher timeframe momentum
# - Volume spike (1.8x 4-day average) confirms institutional participation
# - Works in both bull (buy breakouts in weekly uptrend) and bear (sell breakdowns in weekly downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual weekly data for proper trend alignment without look-ahead
# - Designed for BTC/ETH with focus on quality over quantity
# - Aims for 60-120 total trades over 4 years (15-30/year) to stay within limits