#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses weekly EMA(50) for trend direction to capture major moves in both bull and bear markets.
# Entry when price breaks Donchian(20) high/low with volume spike in trend direction.
# Exit on opposite Donchian(10) break or trend reversal.
# Target: 15-25 trades/year per symbol to minimize fee drag.
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly trend: 50-period EMA on close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Daily volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_ma_20 > 0, volume / vol_ma_20, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(high_10[i]) or 
            np.isnan(low_10[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian(20) high with volume spike in uptrend
            long_condition = (close[i] > high_20[i]) and vol_spike[i] and uptrend
            # Short breakdown: price breaks below Donchian(20) low with volume spike in downtrend
            short_condition = (close[i] < low_20[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian(10) low or trend turns down
            if (close[i] < low_10[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian(10) high or trend turns up
            if (close[i] > high_10[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals