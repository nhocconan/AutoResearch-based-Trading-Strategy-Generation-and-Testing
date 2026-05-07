#!/usr/bin/env python3
name = "12h_Donchian_20_TrendVolume_v1"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian channel (20-day high/low) from daily data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 4)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 20-day high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > high_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below 20-day low with volume and weekly downtrend
            elif close[i] < low_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below 20-day high or volume drops
            if close[i] < high_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above 20-day low or volume drops
            if close[i] > low_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with weekly trend and volume confirmation
# - Donchian(20) from daily data provides key support/resistance levels
# - Breakout above 20-day high with volume in weekly uptrend = long opportunity
# - Breakdown below 20-day low with volume in weekly downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Weekly trend filter reduces whipsaws and aligns with higher timeframe momentum
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to the 20-day channel or volume weakens
# - Position size 0.30 targets ~20-40 trades/year, avoiding excessive fee drag
# - Uses actual daily Donchian levels (not intraday) for proper structure
# - Weekly EMA(34) ensures we only trade in the direction of the higher timeframe trend
# - Volume confirmation reduces false breakouts from low-liquidity moves
# - Designed for 12h timeframe to stay within trade frequency limits (50-150 total over 4 years)