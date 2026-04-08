#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v5
# Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
# Long when price breaks above 4h Donchian high (20), 1-day EMA50 > EMA200 (uptrend), and volume > 1.5x 20-period average.
# Short when price breaks below 4h Donchian low (20), 1-day EMA50 < EMA200 (downtrend), and volume > 1.5x 20-period average.
# Exit when price returns to 4h Donchian midline (average of 20-period high/low).
# Designed for low trade frequency (~25-40/year) with strong edge in both bull and bear markets via trend alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 1-day EMA trend filter (EMA50 vs EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, 200)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian midline
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian midline
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high, 1-day EMA50 > EMA200, volume surge
            if (close[i] > highest_high[i] and 
                ema_50_aligned[i] > ema_200_aligned[i] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, 1-day EMA50 < EMA200, volume surge
            elif (close[i] < lowest_low[i] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals