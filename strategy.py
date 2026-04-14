#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation
# Williams Alligator identifies trend using smoothed medians (Jaw/Teeth/Lips).
# Trend filter: weekly EMA(50) - only long above, short below weekly EMA.
# Volume: current volume > 1.5x 24-period average for confirmation.
# Works in bull/bear by using weekly EMA trend filter.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Alligator and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator components (smoothed medians)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(high_1w, 13)  # Using high for Jaw
    teeth_raw = smma(low_1w, 8)   # Using low for Teeth
    lips_raw = smma(close_1w, 5)  # Using close for Lips
    
    # Shift the smoothed values (Jaw: -8, Teeth: -5, Lips: -3)
    jaw_shifted = np.roll(jaw_raw, 8)
    teeth_shifted = np.roll(teeth_raw, 5)
    lips_shifted = np.roll(lips_raw, 3)
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    
    # Weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13+8, 8+5, 5+3, 24, 50)  # max of all lookbacks
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND above weekly EMA50 AND volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                price > ema_50_1w_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: Jaw > Teeth > Lips (bearish alignment) AND below weekly EMA50 AND volume confirmation
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  price < ema_50_1w_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) OR below weekly EMA50
            if lips_aligned[i] < teeth_aligned[i] or price < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines cross (Jaw < Teeth) OR above weekly EMA50
            if jaw_aligned[i] < teeth_aligned[i] or price > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Alligator_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0