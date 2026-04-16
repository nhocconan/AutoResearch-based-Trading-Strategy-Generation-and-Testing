#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h pivot direction filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 12h pivot shows bullish bias (price > daily pivot) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 12h pivot shows bearish bias (price < daily pivot) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 12h pivot ensures alignment with higher timeframe structure, volume confirms participation.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets with strict entry conditions to limit trades and reduce fee drag.
# Target: 75-175 trades over 4 years (19-44/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_up = highest_high
    donchian_low = lowest_low
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for pivot filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for pivot calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Daily Pivot Points (Classic) ===
    pivot = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    # Bullish bias: price > pivot, Bearish bias: price < pivot
    bullish_bias = close_12h > pivot
    bearish_bias = close_12h < pivot
    
    # Align 12h data to 6h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_12h, donchian_up)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    bullish_bias_aligned = align_htf_to_ltf(prices, df_12h, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_12h, bearish_bias.astype(float))
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian and volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_up_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(bullish_bias_aligned[i]) or np.isnan(bearish_bias_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume[i] > (1.5 * vol_ma_aligned[i])
        bull_bias = bullish_bias_aligned[i] > 0.5
        bear_bias = bearish_bias_aligned[i] > 0.5
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian midpoint or volume spike ends
            midpoint = (donchian_up_aligned[i] + donchian_low_aligned[i]) / 2.0
            if price <= midpoint or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian midpoint or volume spike ends
            midpoint = (donchian_up_aligned[i] + donchian_low_aligned[i]) / 2.0
            if price >= midpoint or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND 12h bullish bias AND volume spike
            if price > donchian_up_aligned[i] and bull_bias and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower band AND 12h bearish bias AND volume spike
            elif price < donchian_low_aligned[i] and bear_bias and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_12hPivotBias_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0