#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA) AND Jaw > Teeth > Lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when price < Alligator Jaw AND Jaw < Teeth < Lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not in proper order) OR price crosses Jaw
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to avoid overtrading.
# Williams Alligator uses smoothed moving averages (SMMA) which reduces whipsaw in choppy markets.
# The 1d EMA50 ensures alignment with higher timeframe trend, preventing counter-trend trades.
# Volume confirmation filters out low-conviction breakouts.
# Works in bull markets by capturing strong uptrends with proper Alligator alignment.
# Works in bear markets by capturing strong downtrends with proper Alligator alignment.
# The SMMA smoothing provides natural filtering compared to regular MA crossovers.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple SMA
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply the Alligator shifts (forward shift = look-ahead, so we need to align properly)
    # Actually, Alligator uses future values, so we need to use lagged values for proper alignment
    # Jaw: 13-period SMMA of close, then shift 8 bars back (use value from 8 bars ago)
    jaw_shifted = np.roll(jaw, 8)  # Shift forward by 8 positions
    jaw_shifted[:8] = np.nan  # First 8 values become NaN
    
    # Teeth: 8-period SMMA of close, then shift 5 bars back
    teeth_shifted = np.roll(teeth, 5)  # Shift forward by 5 positions
    teeth_shifted[:5] = np.nan  # First 5 values become NaN
    
    # Lips: 5-period SMMA of close, then shift 3 bars back
    lips_shifted = np.roll(lips, 3)  # Shift forward by 3 positions
    lips_shifted[:3] = np.nan  # First 3 values become NaN
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 13+8, 8+5, 5+3)  # volume MA, EMA50, and Alligator shifts warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = curr_jaw > curr_teeth > curr_lips
        bearish_alignment = curr_jaw < curr_teeth < curr_lips
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR price crosses below Jaw
            if not bullish_alignment or curr_close < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR price crosses above Jaw
            if not bearish_alignment or curr_close > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price > Jaw AND bullish alignment AND price > 1d EMA50 AND volume confirmation
            if curr_close > curr_jaw and bullish_alignment and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < Jaw AND bearish alignment AND price < 1d EMA50 AND volume confirmation
            elif curr_close < curr_jaw and bearish_alignment and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals