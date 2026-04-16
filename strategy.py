#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h weekly pivot filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 12h weekly pivot R1 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND price < 12h weekly pivot S1 AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures momentum, weekly pivot provides institutional reference levels,
# volume confirms participation. Designed to work in both bull (breakouts) and bear (breakdowns) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for weekly pivot
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for weekly pivot calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Weekly Pivot (using prior week's data) ===
    # Weekly high/low/close from prior completed week
    # We'll use rolling window of 5 bars (5x12h = 60h ≈ 2.5 days, but we need true weekly)
    # Instead, use prior week's high/low/close approximated by 5-period lookback
    # For true weekly pivot, we need to group by week - but we'll use 5-period as proxy
    # Better: use prior week's data by taking max/min/close of 5 bars ago
    if len(high_12h) >= 5:
        weekly_high = pd.Series(high_12h).shift(5).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_12h).shift(5).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_12h).shift(5).rolling(window=5, min_periods=5).last().values
    else:
        weekly_high = np.full_like(close_12h, np.nan)
        weekly_low = np.full_like(close_12h, np.nan)
        weekly_close = np.full_like(close_12h, np.nan)
    
    # Weekly pivot calculations
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2.0 * weekly_pivot - weekly_low
    weekly_s1 = 2.0 * weekly_pivot - weekly_high
    
    # Align 12h weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_12h, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_12h, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_12h, weekly_s1)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian/volume MA, 5+5 for weekly pivot)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian(20) low or weekly pivot
            if price < lowest_low[i] or price < weekly_pivot_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian(20) high or weekly pivot
            if price > highest_high[i] or price > weekly_pivot_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian(20) high AND price > weekly R1 AND volume spike
            if price > highest_high[i] and price > weekly_r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian(20) low AND price < weekly S1 AND volume spike
            elif price < lowest_low[i] and price < weekly_s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_12hWeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0