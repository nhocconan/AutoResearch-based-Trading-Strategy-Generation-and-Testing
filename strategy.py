#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: 4h Donchian channel breakouts in the direction of the daily trend 
with volume confirmation capture sustained moves in both bull and bear markets.
Daily trend filter prevents counter-trend trades, reducing whipsaw in ranging markets.
Volume ensures breakouts have institutional participation.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
Works in bull markets via breakout momentum and in bear markets via breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Donchian channel (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        donchian_high = highest_20[i]
        donchian_low = lowest_20[i]
        daily_trend = ema_50d_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low or close[i] < daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high or close[i] > daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with bullish trend and volume
            if close[i] > donchian_high and close[i] > daily_trend and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with bearish trend and volume
            elif close[i] < donchian_low and close[i] < daily_trend and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals