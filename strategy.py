#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction + 6h Donchian breakout for entry timing
# Long when: price breaks above 6h Donchian Upper(20) AND 6h close > 1d Alligator Lips (smoothed median price) AND Alligator is bullish (Lips > Teeth > Jaw)
# Short when: price breaks below 6h Donchian Lower(20) AND 6h close < 1d Alligator Lips AND Alligator is bearish (Lips < Teeth < Jaw)
# Exit when: price crosses back inside Donchian channel OR Alligator trend reverses
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams Alligator provides smooth trend filter from higher timeframe (1d)
# Donchian breakout captures momentum entries with proper timing
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "6h_WilliamsAlligator_1dTrend_DonchianBreakout_6h"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least one completed 1d bar for SMAs
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Alligator: (H+L+C)/3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Williams Alligator lines (all SMAs with specific periods and shifts)
    # Jaw: SMA(13) of typical price, shifted 8 bars forward
    # Teeth: SMA(8) of typical price, shifted 5 bars forward  
    # Lips: SMA(5) of typical price, shifted 3 bars forward
    jaw_1d = pd.Series(typical_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(typical_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(typical_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1d Alligator lines to 6h timeframe (wait for completed 1d bar)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian channels (20-period)
    upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 6h Donchian channels to 6h timeframe (no additional delay needed for breakout)
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(upper_6h_aligned[i]) or 
            np.isnan(lower_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 6h Donchian Upper, above 1d Alligator Lips, Alligator bullish
            if (close[i] > upper_6h_aligned[i] and 
                close[i] > lips_1d_aligned[i] and 
                lips_1d_aligned[i] > teeth_1d_aligned[i] and 
                teeth_1d_aligned[i] > jaw_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h Donchian Lower, below 1d Alligator Lips, Alligator bearish
            elif (close[i] < lower_6h_aligned[i] and 
                  close[i] < lips_1d_aligned[i] and 
                  lips_1d_aligned[i] < teeth_1d_aligned[i] and 
                  teeth_1d_aligned[i] < jaw_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses back inside Donchian channel OR Alligator turns bearish
            if (close[i] < upper_6h_aligned[i] and close[i] > lower_6h_aligned[i]) or \
               (lips_1d_aligned[i] < teeth_1d_aligned[i] or teeth_1d_aligned[i] < jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses back inside Donchian channel OR Alligator turns bullish
            if (close[i] < upper_6h_aligned[i] and close[i] > lower_6h_aligned[i]) or \
               (lips_1d_aligned[i] > teeth_1d_aligned[i] and teeth_1d_aligned[i] > jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals