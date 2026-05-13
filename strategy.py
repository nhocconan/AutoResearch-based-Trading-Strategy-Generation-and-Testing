#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike.
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND close > 1d EMA50 AND volume > 2.0x 20-period average.
# Short when jaws < teeth < lips AND close < 1d EMA50 AND volume > 2.0x 20-period average.
# Exit on opposite Alligator alignment (jaws < teeth for longs, jaws > teeth for shorts) or ATR trailing stop (2.5x).
# Uses 4h timeframe with 1d trend filter for noise reduction, targeting 75-200 trades over 4 years.
# Williams Alligator identifies trending markets via SMMA alignment, EMA50 filters intermediate trend, volume confirms breakout authenticity.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - Williams Alligator uses SMMA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMMA(5), SMMA(8), SMMA(13) on median price
    median_price = (high + low) / 2
    lips = smma(median_price, 5)   # SMMA(5)
    teeth = smma(median_price, 8)  # SMMA(8)
    jaws = smma(median_price, 13)  # SMMA(13)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 4h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: jaws > teeth > lips (bullish alignment) AND close > 1d EMA50 AND volume spike
            if jaws[i] > teeth[i] and teeth[i] > lips[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: jaws < teeth < lips (bearish alignment) AND close < 1d EMA50 AND volume spike
            elif jaws[i] < teeth[i] and teeth[i] < lips[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: jaws < teeth (bullish alignment broken) OR trailing stop hit
            alignment_exit = jaws[i] < teeth[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if alignment_exit or trailing_stop:
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
            # EXIT SHORT: jaws > teeth (bearish alignment broken) OR trailing stop hit
            alignment_exit = jaws[i] > teeth[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if alignment_exit or trailing_stop:
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