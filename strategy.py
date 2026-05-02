#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Weekly pivot (Camarilla) provides institutional structure on higher timeframe, filtering breakouts
# Volume spike (2.0x 20-period average) ensures strong participation and reduces false breakouts
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring weekly trend alignment and volume confirmation

name = "6h_Donchian20_WeeklyCamarilla_PivotTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3, S3, R4, S4)
    # Based on previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_val = weekly_high - weekly_low
    
    R3 = pivot + range_val * 1.1 / 2
    S3 = pivot - range_val * 1.1 / 2
    R4 = pivot + range_val * 1.1
    S4 = pivot - range_val * 1.1
    
    # Weekly trend: price > R3 = bullish, price < S3 = bearish
    weekly_trend_bullish = weekly_close > R3
    weekly_trend_bearish = weekly_close < S3
    
    # Align weekly indicators to 6h timeframe
    weekly_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bullish.astype(float))
    weekly_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, weekly data and volume MA)
    start_idx = 55  # max(20 for Donchian/volume) + buffer for weekly alignment
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_trend_bullish_aligned[i]) or np.isnan(weekly_trend_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper channel + weekly bullish trend + volume spike
            if close[i] > highest_high[i] and weekly_trend_bullish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel + weekly bearish trend + volume spike
            elif close[i] < lowest_low[i] and weekly_trend_bearish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to midpoint of Donchian channel
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to midpoint of Donchian channel
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals