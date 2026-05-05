#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams Alligator (JAW/TEETH/LIPS) with Donchian(20) breakout and volume confirmation
# Long when price breaks above Donchian(20) upper band AND Alligator is bullish (LIPS > TEETH > JAW) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) lower band AND Alligator is bearish (LIPS < TEETH < JAW) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses Donchian(20) middle band (20-period SMA of high/low) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trend alignment (bullish/bearish) to avoid chop
# Donchian(20) provides clear breakout levels with built-in trend filter
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with bullish Alligator) and bear markets (breakdowns with bearish Alligator)

name = "12h_WilliamsAlligator_Donchian20_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

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
    if len(df_1d) < 13:  # Need at least 13 for Alligator (8+5+3)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2.0  # Typical price for Alligator
    
    # Calculate Williams Alligator: JAW(13,8), TEETH(8,5), LIPS(5,3)
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Align daily Alligator to 12h timeframe (wait for completed daily bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Donchian(20) on 12h data
    def donchian_bands(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = donchian_bands(high, low, 20)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper, Alligator bullish (Lips > Teeth > Jaw), volume confirmation, in session
            if (close[i] > donchian_upper[i] and 
                lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower, Alligator bearish (Lips < Teeth < Jaw), volume confirmation, in session
            elif (close[i] < donchian_lower[i] and 
                  lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian middle OR volume drops below average
            if close[i] < donchian_middle[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian middle OR volume drops below average
            if close[i] > donchian_middle[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals