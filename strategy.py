#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_Confirm
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 with volume confirmation and price > daily close; enter short when price breaks below S1 with volume confirmation and price < daily close. Uses 1d close trend filter to avoid counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Camarilla Pivot Levels (based on previous 12h bar) ===
    # Calculate pivot points using previous bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # Avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # === 1d close trend filter (bullish/bearish daily candle) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    daily_bullish = close_1d > open_1d  # Bullish daily candle
    daily_bearish = close_1d < open_1d  # Bearish daily candle
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # === 1d volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20  # For volume average
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or
            np.isnan(daily_bullish_aligned[i]) or
            np.isnan(daily_bearish_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Daily trend filters
        daily_bull = daily_bullish_aligned[i]
        daily_bear = daily_bearish_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 + volume filter + bullish daily candle
            if close[i] > r1[i] and vol_filter and daily_bull:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 + volume filter + bearish daily candle
            elif close[i] < s1[i] and vol_filter and daily_bear:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price closes below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_Confirm"
timeframe = "12h"
leverage = 1.0