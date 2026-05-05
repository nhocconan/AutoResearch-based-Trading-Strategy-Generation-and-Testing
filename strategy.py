#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian upper band AND weekly pivot shows bullish bias AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND weekly pivot shows bearish bias AND volume > 1.5x 20-period average
# Exit when price reverts to Donchian middle band OR weekly pivot bias flips
# Weekly pivot provides structural bias from higher timeframe, Donchian captures breakouts, volume filter avoids false signals
# Works in bull markets via longs in bullish weekly bias and bear markets via shorts in bearish weekly bias

name = "6h_Donchian20_WeeklyPivot_Bias_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    prev_week_high = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
    prev_week_low = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
    prev_week_close = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    support_1 = (2 * pivot_point) - prev_week_high
    resistance_1 = (2 * pivot_point) - prev_week_low
    
    # Weekly bias: bullish if close > resistance_1, bearish if close < support_1
    weekly_bullish = prev_week_close > resistance_1
    weekly_bearish = prev_week_close < support_1
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian channels (20-period) on 6h data
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND weekly bullish bias AND volume spike
            if (close[i] > donchian_upper[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND weekly bearish bias AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Donchian middle OR weekly bias turns bearish
            if (close[i] < donchian_middle[i] or 
                weekly_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Donchian middle OR weekly bias turns bullish
            if (close[i] > donchian_middle[i] or 
                weekly_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals