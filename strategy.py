#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot bias is bullish AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND weekly pivot bias is bearish AND volume > 1.5x 20-bar avg
# Exit when price crosses Donchian(20) midpoint (mean reversion structure)
# Weekly pivot bias: price above weekly Camarilla pivot (PP) = bullish, below = bearish
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 6h.
# Donchian breakouts capture momentum; weekly pivot filters counter-trend moves; volume confirms conviction.
# Works in both bull and bear markets by aligning with higher timeframe structure.

name = "6h_Donchian20_1dWeeklyPivot_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data (using last 5 daily bars ≈ 1 week)
    # We need at least 5 daily bars to calculate weekly OHLC
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get last 5 daily bars for weekly calculation
    weekly_high = np.max(df_1d['high'].iloc[-5:].values)
    weekly_low = np.min(df_1d['low'].iloc[-5:].values)
    weekly_close = df_1d['close'].iloc[-1]
    
    # Calculate weekly pivot (PP) and support/resistance levels
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pp - weekly_low
    weekly_s1 = 2 * weekly_pp - weekly_high
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    weekly_r3 = weekly_r1 + (weekly_high - weekly_low)
    weekly_s3 = weekly_s1 - (weekly_high - weekly_low)
    
    # Weekly pivot bias: price above PP = bullish, below = bearish
    weekly_bullish = weekly_close > weekly_pp
    weekly_bearish = weekly_close < weekly_pp
    
    # Align weekly bias to 6h timeframe (same bias for all 6h bars within the week)
    # Create array of same length as df_1d with the bias value
    weekly_bullish_array = np.full(len(df_1d), weekly_bullish, dtype=bool)
    weekly_bearish_array = np.full(len(df_1d), weekly_bearish, dtype=bool)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish_array.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish_array.astype(float))
    
    # Donchian(20) calculation on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian needs 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND weekly bullish AND volume confirmation
            if curr_close > dh and weekly_bull and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND weekly bearish AND volume confirmation
            elif curr_close < dl and weekly_bear and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses Donchian midpoint (mean reversion)
            if curr_close < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses Donchian midpoint (mean reversion)
            if curr_close > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals