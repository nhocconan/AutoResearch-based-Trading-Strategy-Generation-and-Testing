#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# combined with 4h Donchian channel breakout for entry timing and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND Alligator is bullish (Lips > Teeth > Jaw)
# Short when price breaks below Donchian(20) lower band AND Alligator is bearish (Lips < Teeth < Jaw)
# Exit when price crosses Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk while minimizing fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Alligator provides smooth trend filter with built-in smoothing
# Donchian breakout captures momentum with clear structure
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_WilliamsAlligator_Donchian20_VolumeConfirm"
timeframe = "4h"
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
    if len(df_1d) < 13:  # Need enough for Alligator (13,8,5)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    # All lines are smoothed with future values (so we need to align properly)
    close_1d_series = pd.Series(close_1d)
    jaw = close_1d_series.ewm(span=13, adjust=False).mean().values  # Jaw (Blue)
    teeth = close_1d_series.ewm(span=8, adjust=False).mean().values   # Teeth (Red)
    lips = close_1d_series.ewm(span=5, adjust=False).mean().values    # Lips (Green)
    
    # Align Alligator lines to 4h timeframe (wait for completed daily bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 4h data ONCE before loop for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator trend conditions
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND Alligator bullish AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and bullish_alligator and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Donchian lower AND Alligator bearish AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and bearish_alligator and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian mid OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price crosses above Donchian mid OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals