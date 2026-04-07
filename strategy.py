#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Daily trend + 12h Donchian breakout with volume confirmation captures strong momentum moves.
In bull markets, buy breakouts above 20-period high; in bear markets, sell breakdowns below 20-period low.
Daily EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
Volume confirmation reduces false breakouts. Target: 15-25 trades/year on 12h to minimize fee drag.
Works in both bull and bear markets by following daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    
    # Align daily EMA50 to 12h timeframe
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Donchian channels (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        daily_trend = ema_50d_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low or trend turns bearish
            if low[i] < low_20[i] or close[i] < daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high or trend turns bullish
            if high[i] > high_20[i] or close[i] > daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-period high with bullish trend and volume
            if high[i] > high_20[i] and close[i] > daily_trend and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-period low with bearish trend and volume
            elif low[i] < low_20[i] and close[i] < daily_trend and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals