#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (teeth > lips) AND 1d bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when price < Alligator Jaw (teeth < lips) AND 1d bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Williams Alligator (Smoothed Medians): Jaw=13, Teeth=8, Lips=5 (all shifted forward)
# Uses 1d EMA50 for trend filter to reduce whipsaw, targeting 12-37 trades/year on 12h.
# Volume confirmation (1.5x) and Alligator alignment reduce noise trades. Works in ranging and trending markets.
# Alligator provides dynamic support/resistance; trend filter ensures direction alignment.

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Williams Alligator: Smoothed Medians (SMMA)
    def smma(arr, period):
        """Smoothed Moving Average (Williams Alligator uses SMMA)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines: Jaw (13, shifted 8), Teeth (8, shifted 5), Lips (5, shifted 3)
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift forward: Jaw by 8, Teeth by 5, Lips by 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate shifted values (set to NaN where roll created invalid data)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator condition: Jaw > Teeth > Lips (bullish alignment) OR Jaw < Teeth < Lips (bearish alignment)
    # We use the Jaw as the main reference line for price comparison
    alligator_bullish = (jaw > teeth) & (teeth > lips)
    alligator_bearish = (jaw < teeth) & (teeth < lips)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Alligator bullish alignment AND 1d bullish trend AND volume spike
            if (close[i] > jaw[i] and 
                alligator_bullish[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND Alligator bearish alignment AND 1d bearish trend AND volume spike
            elif (close[i] < jaw[i] and 
                  alligator_bearish[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Jaw OR Alligator turns bearish OR 1d trend turns bearish
            if (close[i] < jaw[i] or 
                not alligator_bullish[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Jaw OR Alligator turns bullish OR 1d trend turns bullish
            if (close[i] > jaw[i] or 
                not alligator_bearish[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals