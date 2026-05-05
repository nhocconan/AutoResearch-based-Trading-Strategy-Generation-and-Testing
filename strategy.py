#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams Alligator (SMMA) trend filter + Donchian(20) breakout + volume spike confirmation
# Long when price breaks above Donchian(20) high AND price > Alligator Jaw (SMMA13) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND price < Alligator Jaw (SMMA13) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above Donchian(20) midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator provides smooth trend identification using SMMA (Smoothed Moving Average)
# Donchian(20) gives clear breakout levels with built-in volatility adaptation
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_WilliamsAlligator_Donchian20_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's EMA with alpha=1/period"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for Alligator calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Alligator calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Williams Alligator: Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5)
    jaw_1d = smma(typical_price_1d, 13)
    teeth_1d = smma(typical_price_1d, 8)
    lips_1d = smma(typical_price_1d, 5)
    
    # Align daily Alligator to 12h timeframe (wait for completed daily bar)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate Donchian(20) on 12h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, above Alligator Jaw, volume confirmation, in session
            if close[i] > highest_high[i] and close[i] > jaw_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, below Alligator Jaw, volume confirmation, in session
            elif close[i] < lowest_low[i] and close[i] < jaw_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals