#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when: price above Alligator jaws (green line) + 1d EMA(50) rising + volume spike
# Short when: price below Alligator jaws (red line) + 1d EMA(50) falling + volume spike
# Exit when: price crosses Alligator teeth (red line) or trend reverses
# Williams Alligator: Jaw (13-period SMMA, 8 bars forward), Teeth (8-period SMMA, 5 bars forward), Lips (5-period SMMA, 3 bars forward)
# Uses Williams Alligator to identify trend direction and entry, with 1d EMA for higher timeframe trend filter
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 15-30 trades/year on 12h timeframe.
# Designed to work in both bull (trend following) and bear (trend following on shorts) markets.

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also known as Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(data)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(median_price_12h, 13)  # Blue line
    teeth = smma(median_price_12h, 8)  # Red line
    lips = smma(median_price_12h, 5)   # Green line
    
    # Apply forward shift as per Williams Alligator definition
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (they're already on 12h, but ensure proper alignment)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above lips (green) + 1d EMA rising + volume spike
            # In Alligator, when lips > teeth > jaw (green above red above blue) = uptrend
            if (close[i] > lips_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below lips (green) + 1d EMA falling + volume spike
            # In Alligator, when lips < teeth < jaw (green below red below blue) = downtrend
            elif (close[i] < lips_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below teeth (red) OR trend turns down
            if (close[i] < teeth_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above teeth (red) OR trend turns up
            if (close[i] > teeth_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals