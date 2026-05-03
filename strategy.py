#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray combination with volume confirmation.
# Long when: Alligator jaw < teeth < lips (bullish alignment) AND Bear Power > 0 (bullish momentum) AND 6h volume > 1.5x 20-period volume MA.
# Short when: Alligator jaw > teeth > lips (bearish alignment) AND Bull Power < 0 (bearish momentum) AND 6h volume > 1.5x 20-period volume MA.
# Exit when Alligator alignment breaks OR Elder Power reverses sign.
# Uses 6h timeframe to target 50-150 total trades over 4 years (12-37/year) with strict multi-condition entry.
# Williams Alligator identifies trend structure via smoothed medians, Elder Ray measures bull/bear power relative to EMA13,
# Volume confirms participation. Works in both bull and bear markets by requiring alignment of trend, momentum, and volume.

name = "6h_Alligator_ElderRay_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Elder Ray (Bull/Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams Alligator from 6h data (smoothed medians)
    # Jaw: 13-period SMMA smoothed 8 bars -> 21
    # Teeth: 8-period SMMA smoothed 5 bars -> 13
    # Lips: 5-period SMMA smoothed 3 bars -> 8
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply smoothing offsets: Jaw +8, Teeth +5, Lips +3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # First values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h (already on 6h, but ensure alignment for safety)
    jaw_aligned = jaw  # Already on 6h
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Alligator conditions
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Elder Ray conditions
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.5)
        
        if position == 0:
            # Long: Bullish Alligator alignment AND Bullish Elder Ray AND volume spike AND session
            if bullish_alignment and bullish_momentum and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND Bearish Elder Ray AND volume spike AND session
            elif bearish_alignment and bearish_momentum and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Elder Ray turns bearish
            if not bullish_alignment or not bullish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Elder Ray turns bullish
            if not bearish_alignment or not bearish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals