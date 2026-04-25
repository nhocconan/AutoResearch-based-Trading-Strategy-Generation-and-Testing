#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Confluence
Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Weekly pivot provides structural bias (bullish/bearish/neutral) from higher timeframe.
Donchian breakout captures momentum in direction of weekly bias.
Volume confirmation filters false breakouts.
Designed for low frequency (12-30 trades/year) to minimize fee drag on 6h timeframe.
Works in bull/bear by following weekly structural bias, avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for weekly pivot calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Need enough 1d data to compute weekly pivot (approx 5 trading days)
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC (using last 5 daily bars)
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows  
    # Weekly close = last daily close
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = df_1d['close'].shift(1).values  # prior week's close
    
    # Weekly pivot points (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Bias determination: price vs weekly pivot and support/resistance
    # Bullish bias: price above weekly pivot and above weekly S1
    # Bearish bias: price below weekly pivot and below weekly R1
    # Neutral: otherwise
    bullish_bias = (weekly_close > weekly_pivot) & (weekly_close > weekly_s1)
    bearish_bias = (weekly_close < weekly_pivot) & (weekly_close < weekly_r1)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias.astype(float))
    
    # 6h data for Donchian(20) breakout (using current timeframe)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly calculation (5d) + donchian (20) + volume MA (20)
    start_idx = max(5, donchian_window, 20) + 5  # conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Get bias signals (already aligned)
        bullish_bias_signal = bullish_bias_aligned[i] > 0.5
        bearish_bias_signal = bearish_bias_aligned[i] > 0.5
        
        if position == 0:
            # Look for entry signals with volume confirmation and weekly bias alignment
            # Long breakout: price breaks above Donchian high with bullish bias and volume
            long_breakout = (curr_high > donchian_high[i]) and bullish_bias_signal and volume_confirm[i]
            # Short breakout: price breaks below Donchian low with bearish bias and volume
            short_breakout = (curr_low < donchian_low[i]) and bearish_bias_signal and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below Donchian low or bias turns bearish
            if curr_low < donchian_low[i] or bearish_bias_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian high or bias turns bullish
            if curr_high > donchian_high[i] or bullish_bias_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Confluence"
timeframe = "6h"
leverage = 1.0