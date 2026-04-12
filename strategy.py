#!/usr/bin/env python3
"""
6h_1d_RVI_Trend_Filtered_Breakout_v1
Hypothesis: On 6h timeframe, use Relative Vigor Index (RVI) from daily timeframe to filter breakouts. Enter long when price breaks above 6h Donchian(20) with RVI > 0.5 and volume spike, enter short when price breaks below Donchian(20) with RVI < -0.5 and volume spike. RVI measures trend strength by comparing close-open to high-low range, providing cleaner trend detection than ADX in choppy markets. Volume confirmation ensures breakout validity. Designed for 15-30 trades/year by requiring multiple confirmations, reducing false breakouts in ranging markets while capturing strong trends in both bull and bear phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RVI_Trend_Filtered_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20 period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data ONCE for RVI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RVI (Relative Vigor Index) on daily timeframe
    # RVI = (Close - Open) / (High - Low) smoothed
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    numerator = close_1d - open_1d
    denominator = high_1d - low_1d
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    raw_rvi = numerator / denominator
    
    # Smooth RVI with 10-period SMA (standard setting)
    rvi = pd.Series(raw_rvi).rolling(window=10, min_periods=10).mean().values
    
    # Align RVI to 6h timeframe
    rvi_aligned = align_htf_to_ltf(prices, df_1d, rvi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rvi_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]   # Break above upper band
        breakout_down = close[i] < lowest_low[i]   # Break below lower band
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # RVI trend filter: RVI > 0.5 for uptrend, RVI < -0.5 for downtrend
        rvi_uptrend = rvi_aligned[i] > 0.5
        rvi_downtrend = rvi_aligned[i] < -0.5
        
        # Entry conditions: breakout + volume + RVI trend filter
        long_entry = breakout_up and volume_spike and rvi_uptrend
        short_entry = breakout_down and volume_spike and rvi_downtrend
        
        # Exit conditions: price returns to opposite Donchian level or RVI reverses
        long_exit = (close[i] < lowest_low[i]) or (rvi_aligned[i] < 0)
        short_exit = (close[i] > highest_high[i]) or (rvi_aligned[i] > 0)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals