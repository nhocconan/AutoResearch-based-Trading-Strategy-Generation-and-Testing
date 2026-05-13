#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d EMA34 trend + volume confirmation.
# Long when price > Alligator Jaw (teeth > lips) AND 1d EMA34 rising AND volume > 1.5x average.
# Short when price < Alligator Jaw (teeth < lips) AND 1d EMA34 falling AND volume > 1.5x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Alligator identifies trend via smoothed medians (13,8,5), 1d EMA34 filters higher timeframe trend,
# volume confirms breakout strength. Designed for 6h to capture medium-term swings in both bull/bear.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: Smoothed medians (Jaw=13, Teeth=8, Lips=5)
    # Smoothed median = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean()  # Additional smoothing
    jaw = jaw.rolling(window=5, min_periods=5).mean()
    jaw_values = jaw.values
    
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean()
    teeth = teeth.rolling(window=3, min_periods=3).mean()
    teeth_values = teeth.values
    
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean()
    lips = lips.rolling(window=3, min_periods=3).mean()
    lips_values = lips.values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Jaw AND Teeth > Lips (bullish alignment) AND 1d EMA34 rising AND volume > 1.5x average
            if (close[i] > jaw_values[i] and 
                teeth_values[i] > lips_values[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Jaw AND Teeth < Lips (bearish alignment) AND 1d EMA34 falling AND volume > 1.5x average
            elif (close[i] < jaw_values[i] and 
                  teeth_values[i] < lips_values[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals