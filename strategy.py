#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume spike
# Long when: price > Alligator Jaw (13-period SMMA shifted 8) + 1d EMA(50) rising + volume spike (>2x 20-period avg)
# Short when: price < Alligator Teeth (8-period SMMA shifted 5) + 1d EMA(50) falling + volume spike
# Exit when: price crosses Alligator Lips (5-period SMMA shifted 3) OR trend reverses
# Position size: 0.25 to limit drawdown. Target: 12-37 trades/year on 12h.
# Williams Alligator catches trends early; EMA filter avoids counter-trend trades; volume confirms conviction.

name = "12h_WilliamsAlligator_1dEMA_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(src, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Smoothing"""
    if length <= 0:
        return src.copy()
    result = np.full_like(src, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.mean(src[:length])
    # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current Price) / length
    for i in range(length, len(src)):
        result[i] = (result[i-1] * (length-1) + src[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (13/8/5 periods, shifted 8/5/3)
    jaw = smma(high, 13)  # Jaw: 13-period SMMA of High
    teeth = smma(low, 8)   # Teeth: 8-period SMMA of Low
    lips = smma(close, 5)  # Lips: 5-period SMMA of Close
    
    # Apply shifts (Jaw shifted 8 bars, Teeth 5 bars, Lips 3 bars)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for validity
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 2.0x 20-period average volume (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or
            np.isnan(lips_shifted[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Jaw + 1d EMA rising + volume spike
            if (close[i] > jaw_shifted[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Teeth + 1d EMA falling + volume spike
            elif (close[i] < teeth_shifted[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Lips OR trend turns down
            if (close[i] < lips_shifted[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Lips OR trend turns up
            if (close[i] > lips_shifted[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals