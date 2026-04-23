#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
Long when price > Alligator Jaw (13-period SMMA) AND Alligator Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 rising AND 12h volume > 1.5x 20-period MA.
Short when price < Alligator Jaw AND Alligator Lips < Teeth < Jaw (bearish alignment) AND 1d EMA50 falling AND 12h volume > 1.5x 20-period MA.
Exit when Alligator alignment breaks or 1d EMA50 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator identifies trend presence and direction, 1d EMA50 filters major trend, volume confirmation avoids low-momentum signals.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    alpha = 1.0 / length
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nansum(source[:length]) / length
        for i in range(length, len(source)):
            if not np.isnan(source[i]):
                result[i] = (source[i] + (length - 1) * result[i-1]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (12h timeframe)
    jaw_length = 13   # Jaw (Blue) - 13-period SMMA
    teeth_length = 8  # Teeth (Red) - 8-period SMMA
    lips_length = 5   # Lips (Green) - 5-period SMMA
    
    jaw = smma(close, jaw_length)
    teeth = smma(close, teeth_length)
    lips = smma(close, lips_length)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_length, teeth_length, lips_length, 50, 20) - 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Price > Jaw AND bullish alignment AND EMA50 rising AND volume filter
            if price > jaw_val and bullish_alignment and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price < Jaw AND bearish alignment AND EMA50 falling AND volume filter
            elif price < jaw_val and bearish_alignment and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR EMA50 starts falling
                if not bullish_alignment or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR EMA50 starts rising
                if not bearish_alignment or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0