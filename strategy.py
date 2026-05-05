#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams Alligator (Jaw/Teeth/Lips) for trend direction, combined with 4h Donchian breakout and volume confirmation
# Long when price breaks above 4h Donchian(20) upper band AND Alligator is bullish (Lips > Teeth > Jaw) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian(20) lower band AND Alligator is bearish (Lips < Teeth < Jaw) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above 4h Donchian(20) midpoint OR Alligator trend reverses
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Alligator provides smoothed trend filter to avoid whipsaws
# Donchian breakout captures momentum with clear structure
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_WilliamsAlligator_DonchianBreakout_VolumeConfirm"
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
    if len(df_1d) < 34:  # Need enough for Alligator (13,8,5 smoothed)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (Smoothed Moving Average = SMA with period)
    # Jaw: 13-period SMMA of Median Price, smoothed 8 bars ahead
    # Teeth: 8-period SMMA of Median Price, smoothed 5 bars ahead  
    # Lips: 5-period SMMA of Median Price, smoothed 3 bars ahead
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Jaw: SMA(13) then smoothed by 8
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: SMA(8) then smoothed by 5
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.rolling(window=5, min_periods=5).mean().values
    
    # Lips: SMA(5) then smoothed by 3
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.rolling(window=3, min_periods=3).mean().values
    
    # Align Williams Alligator to 4h timeframe (wait for completed daily bar)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Get 4h data ONCE before loop for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_4h > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high, Alligator bullish (Lips > Teeth > Jaw), volume confirmation, in session
            if (close[i] > donchian_high[i] and 
                lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian low, Alligator bearish (Lips < Teeth < Jaw), volume confirmation, in session
            elif (close[i] < donchian_low[i] and 
                  lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian mid OR Alligator turns bearish
            if (close[i] < donchian_mid[i] or 
                not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian mid OR Alligator turns bullish
            if (close[i] > donchian_mid[i] or 
                not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals