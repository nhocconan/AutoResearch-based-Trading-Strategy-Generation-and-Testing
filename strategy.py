#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND price > 1d EMA50 AND volume > 1.3x 20-period average.
Short when Alligator jaws crosses below teeth AND price < 1d EMA50 AND volume > 1.3x 20-period average.
Exit when Alligator jaws crosses back below teeth (long) or above teeth (short) OR ATR trailing stop (2.0*ATR from extreme).
Williams Alligator identifies trend initiation with smoothing to reduce whipsaws. 4h timeframe balances trade frequency and reliability. Volume confirmation filters weak signals. 1d EMA50 ensures alignment with higher timeframe trend.
Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h: Jaws (13 SMMA of median price), Teeth (8 SMMA), Lips (5 SMMA)
    median_price = (high + low) / 2.0
    jaws = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line (not used for entry but confirms)
    
    # ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20, 14)  # EMA50 needs 50, jaws needs 13, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(jaws[i]) or np.isnan(teeth[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        jaw_val = jaws[i]
        teeth_val = teeth[i]
        
        if position == 0:
            # Bullish crossover: jaws crosses above teeth AND uptrend (price > EMA50) AND volume confirmation
            if jaw_val > teeth_val and jaws[i-1] <= teeth[i-1] and price > ema50_val and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Bearish crossover: jaws crosses below teeth AND downtrend (price < EMA50) AND volume confirmation
            elif jaw_val < teeth_val and jaws[i-1] >= teeth[i-1] and price < ema50_val and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Jaws crosses back below teeth (long) or above teeth (short)
            if position == 1 and jaw_val < teeth_val:
                exit_signal = True
            elif position == -1 and jaw_val > teeth_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_JawsTeethCross_EXIT_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0