#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot bias is bullish AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below Donchian(20) low AND weekly pivot bias is bearish AND volume > 2.0 * 20-bar avg volume
# Exit with signal=0 when price reverses back inside the Donchian H-L range
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Donchian channels provide clear breakout levels; weekly pivot bias ensures higher-timeframe alignment
# Volume spike confirms institutional participation
# Works in bull via buying strength on upside breakouts aligned with weekly bias, works in bear via selling strength on downside breakdowns

name = "6h_Donchian20_WeeklyPivotBias_VolumeSpike_v1"
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
    
    # Get 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian(20) channels on 6h data
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donchian_high = high_6h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_6h_series.rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Get weekly data ONCE before loop for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_point = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot_point - low_1w
    s1 = 2 * pivot_point - high_1w
    r2 = pivot_point + (high_1w - low_1w)
    s2 = pivot_point - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot_point - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_point)
    
    # Weekly pivot bias: bullish if close above pivot, bearish if below pivot
    weekly_bias_bullish = close_1w > pivot_point
    weekly_bias_bearish = close_1w < pivot_point
    
    # Align weekly bias to 6h timeframe
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_bias_bullish_aligned[i]) or np.isnan(weekly_bias_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with weekly bias and volume filters
            # Long: price breaks above Donchian high AND weekly bullish bias AND volume spike
            if close[i] > donchian_high_aligned[i] and weekly_bias_bullish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly bearish bias AND volume spike
            elif close[i] < donchian_low_aligned[i] and weekly_bias_bearish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverses back inside Donchian range (mean reversion)
            if close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverses back inside Donchian range (mean reversion)
            if close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals