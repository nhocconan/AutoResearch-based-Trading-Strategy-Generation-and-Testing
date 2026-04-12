#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_trend
Hypothesis: 12-hour strategy using daily Donchian channel breakouts with volume confirmation and trend filter.
Only takes long positions when price breaks above 20-day high with above-average volume and uptrend,
and short positions when price breaks below 20-day low with above-average volume and downtrend.
Designed to work in both bull and bear markets by following the higher timeframe trend.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12-period volume average for volume confirmation
    vol_avg_12h = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_ok = volume[i] > vol_avg_12h[i]
        
        # Trend determination: EMA50 direction
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        if uptrend and vol_ok and position != 1:
            # Long: break above Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 1
                signals[i] = 0.25
        elif downtrend and vol_ok and position != -1:
            # Short: break below Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = -1
                signals[i] = -0.25
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (close[i] < donchian_low_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_volume_trend"
timeframe = "12h"
leverage = 1.0