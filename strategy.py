#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h for lower trade frequency (target: 12-37/year) and better signal quality.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) on 12h.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1d EMA34 bullish AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1d EMA34 bearish AND volume spike.
- Volume: Current 12h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Exit: Loss of Alligator alignment OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy combines trend-following (Williams Alligator) with volume confirmation and
daily trend filter to avoid counter-trend trades. Works in both bull and bear markets
by only taking trades in the direction of the 1d trend, with volume spikes confirming
participation. Williams Alligator provides smooth trend identification with built-in
filters against choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
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
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams Alligator on 12h
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Alligator formula)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that don't have data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13+8, 8+5, 5+3)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw AND 1d EMA34 bullish (close > EMA)
                if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) and (close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw AND 1d EMA34 bearish (close < EMA)
                elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) and (close[i] < ema_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: loss of bullish alignment OR loss of volume confirmation
            if not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: loss of bearish alignment OR loss of volume confirmation
            if not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0