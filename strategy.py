#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation.
Long when price > Alligator Jaw (13-period SMMA) AND Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 rising AND volume > 1.5x 20-period MA.
Short when price < Alligator Jaw AND Lips < Teeth < Jaw (bearish alignment) AND 1d EMA50 falling AND volume > 1.5x 20-period MA.
Exit when Alligator lines cross (Lips/Jaw crossover) or 1d EMA50 reverses.
Williams Alligator catches trends early with smoothed moving averages, reducing whipsaw.
1d EMA50 filter ensures we trade with the higher timeframe trend.
Volume confirmation adds momentum validity.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nansum(source[:length]) / length
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_PRICE) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA of median price
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Jaw: 13-period SMMA, shifted 8 bars
    teeth = smma(median_price, 8)   # Teeth: 8-period SMMA, shifted 5 bars
    lips = smma(median_price, 5)    # Lips: 5-period SMMA, shifted 3 bars
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 50, 20)  # Alligator shifts, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
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
                # Long exit: Lips/Jaw crossover (bearish) OR EMA50 starts falling
                lips_jaw_cross = lips_val < jaw_val
                ema_reversal = (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1])
                if lips_jaw_cross or ema_reversal:
                    exit_signal = True
            elif position == -1:
                # Short exit: Lips/Jaw crossover (bullish) OR EMA50 starts rising
                lips_jaw_cross = lips_val > jaw_val
                ema_reversal = (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1])
                if lips_jaw_cross or ema_reversal:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0