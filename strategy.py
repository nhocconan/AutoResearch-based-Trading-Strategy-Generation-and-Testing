#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend absence when lines intertwine.
# Trend entry when price is above/below all lines + aligned with 1w EMA50 + volume spike.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion
# when Alligator "sleeps" (lines converge) and price reverts to mean with volume confirmation.

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: SMAs of median price (HL/2)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    median_price = (high + low) / 2.0
    
    # Jaw (13)
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    # Teeth (8)
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    # Lips (5)
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for Alligator to warm up
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Alligator sleeping: lines are close together (market ranging)
        # Jaw-Teeth distance and Teeth-Lips distance both small
        jaw_teeth_dist = np.abs(jaw[i] - teeth[i])
        teeth_lips_dist = np.abs(teeth[i] - lips[i])
        avg_price = (high[i] + low[i]) / 2.0
        # Normalize by average price to make it adaptive
        jaw_teeth_norm = jaw_teeth_dist / avg_price if avg_price > 0 else 0
        teeth_lips_norm = teeth_lips_dist / avg_price if avg_price > 0 else 0
        alligator_sleeping = (jaw_teeth_norm < 0.01) and (teeth_lips_norm < 0.01)  # 1% threshold
        
        if position == 0:
            # Alligator awake (trending) + volume spike + trend alignment
            # Long: price above all lines (Lips > Teeth > Jaw) + bullish trend + volume spike
            # Short: price below all lines (Jaw > Teeth > Lips) + bearish trend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bullish_trend and volume_spike and not alligator_sleeping):
                signals[i] = 0.25
                position = 1
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bearish_trend and volume_spike and not alligator_sleeping):
                signals[i] = -0.25
                position = -1
            # Mean reversion when Alligator sleeping: price extreme + volume spike
            # Long: price below Lips (oversold) + volume spike
            # Short: price above Lips (overbought) + volume spike
            elif alligator_sleeping and volume_spike:
                if close[i] < lips[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif close[i] > lips[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Alligator sleeping AND price crosses back above Teeth (mean reversion)
            # OR trend changes to bearish
            if (alligator_sleeping and close[i] > teeth[i]) or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping AND price crosses back below Teeth (mean reversion)
            # OR trend changes to bullish
            if (alligator_sleeping and close[i] < teeth[i]) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals