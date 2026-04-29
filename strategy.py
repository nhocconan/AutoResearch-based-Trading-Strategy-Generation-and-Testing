#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d weekly pivot R1 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND price < 1d weekly pivot S1 AND volume > 1.5x 20-bar avg
# Exit when price retouches Donchian midpoint (mean reversion) or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 6h.
# Weekly pivot from 1d provides institutional reference points; Donchian breakout captures momentum.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.

name = "6h_Donchian20_1dWeeklyPivot_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data for weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d OHLC (using prior week's data)
    # Weekly high = max of prior 5 trading days' high
    # Weekly low = min of prior 5 trading days' low
    # Weekly close = prior 5th day's close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window of 5 days for weekly OHLC
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # Shift to avoid look-ahead
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2 * pivot - weekly_low
    # S1 = 2 * pivot - weekly_high
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate Donchian(20) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values  # Shift for completed bar
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian(20) and volume MA(20) need 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND close > weekly R1 AND volume confirmation
            if curr_high > dh and curr_close > r1 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND close < weekly S1 AND volume confirmation
            elif curr_low < dl and curr_close < s1 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Donchian midpoint or breaks below Donchian low
            if curr_close <= dm or curr_low < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches Donchian midpoint or breaks above Donchian high
            if curr_close >= dm or curr_high > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals