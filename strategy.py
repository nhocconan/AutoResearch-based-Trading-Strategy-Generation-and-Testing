#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian upper band AND weekly close > weekly pivot (bullish bias) AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian lower band AND weekly close < weekly pivot (bearish bias) AND volume > 1.5x 20 EMA
# Uses 6h for primary timeframe, 1w for trend filter to avoid counter-trend trades, volume for confirmation.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via longs in bullish weeks and bear markets via shorts in bearish weeks.

name = "6h_Donchian20_1wPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot and trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard formula: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Weekly trend: bullish when close > pivot, bearish when close < pivot
    weekly_bullish = close_1w > weekly_pivot
    weekly_bearish = close_1w < weekly_pivot
    
    # Align weekly data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    # We need at least 20 periods for Donchian calculation
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper_band[i] = np.max(high[i-lookback:i])
        lower_band[i] = np.min(low[i-lookback:i])
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND weekly bullish AND volume spike
            if (close[i] > upper_band[i] and 
                bullish_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND weekly bearish AND volume spike
            elif (close[i] < lower_band[i] and 
                  bearish_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band OR weekly turns bearish
            if (close[i] < lower_band[i] or 
                bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band OR weekly turns bullish
            if (close[i] > upper_band[i] or 
                bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals