#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Alligator calculation (based on daily OHLC) and volume/chop filters.
- Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period).
- Entry: Long when Lips > Teeth > Jaw AND volume > 1.5 * 20-period average volume AND CHOP(14) > 61.8 (ranging market).
         Short when Lips < Teeth < Jaw AND volume > 1.5 * 20-period average volume AND CHOP(14) > 61.8.
- Exit: Opposite Alligator alignment (Lips crosses below Teeth for longs, above Teeth for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator catches trends after ranging periods; volume confirms legitimacy; chop filter ensures we're in ranging conditions where Alligator excels.
- Works in both bull and bear markets as it captures trend emergence from consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def median_price(high, low):
    """Calculate median price (typical price without close)."""
    return (high + low) / 2.0

def smma(data, period):
    """Smoothed Moving Average (SmMA) - Wilder's smoothing."""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SmMA(i) = (SmMA(i-1) * (period-1) + data[i]) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def williams_alligator(high, low, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines."""
    med = median_price(high, low)
    jaw = smma(med, jaw_period)
    teeth = smma(med, teeth_period)
    lips = smma(med, lips_period)
    return jaw, teeth, lips

def choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index."""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_series.rolling(window=period, min_periods=period).max()
    lowest_low = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Alligator
        return np.zeros(n)
    
    jaw_1d, teeth_1d, lips_1d = williams_alligator(
        df_1d['high'].values, 
        df_1d['low'].values
    )
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    chop_1d = choppiness_index(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need 20 for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: Alligator alignment breaks (Lips crosses Teeth)
        if position != 0:
            # Exit long: Lips crosses below Teeth
            if position == 1:
                if lips_1d_aligned[i] < teeth_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Lips crosses above Teeth
            elif position == -1:
                if lips_1d_aligned[i] > teeth_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volume confirmation and chop filter
        if position == 0:
            # Alligator alignment signals
            bullish_alignment = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
            bearish_alignment = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Choppiness regime filter: CHOP > 61.8 (ranging market)
            chop_regime = chop_1d_aligned[i] > 61.8
            
            if bullish_alignment and volume_confirm and chop_regime:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and volume_confirm and chop_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0