#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 bullish AND volume > 1.5 * volume MA(30).
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1d EMA50 bearish AND volume > 1.5 * volume MA(30).
- Exit: Close-based reversal - exit when Alligator alignment breaks (Lips crosses Teeth) OR 1d EMA50 trend changes.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses Williams Alligator to identify trending markets with reduced whipsaw in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA shifted by 8 bars
    jaw_raw = smma(close, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (look back)
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted by 5 bars
    teeth_raw = smma(close, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5 (look back)
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA shifted by 3 bars
    lips_raw = smma(close, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3 (look back)
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume MA(30) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for SMMA and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish alignment AND 1d EMA50 bullish AND volume confirmed
            if bullish_alignment and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND 1d EMA50 bearish AND volume confirmed
            elif bearish_alignment and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Alligator alignment breaks OR trend changes
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if not bullish_alignment or curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Alligator alignment breaks OR trend changes
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            if not bearish_alignment or curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0