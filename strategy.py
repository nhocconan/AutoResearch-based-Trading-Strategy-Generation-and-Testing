#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d chop regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for chop regime detection (Choppiness Index) and volume spike filter.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Regime: Chop > 61.8 = ranging (fade Alligator alignment), Chop < 38.2 = trending (trade Alligator alignment).
- Entry: Long when Lips > Teeth > Jaw AND trending regime AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw AND trending regime AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips crosses Teeth) OR chop regime shifts to ranging (Chop > 61.8).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading strong Alligator alignment in trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.empty_like(series, dtype=float)
    result[:] = np.nan
    # First value is SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Choppiness Index for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Chop calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(14) for Chop denominator
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    denominator = hh14 - ll14
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(sum_atr14 / denominator) / np.log10(14)
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h Williams Alligator
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(close, 13)
    jaw = smma(jaw_raw, 8)
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(close, 8)
    teeth = smma(teeth_raw, 5)
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(close, 5)
    lips = smma(lips_raw, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough for all smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: Chop < 38.2 = trending (favor Alligator alignment trades)
        trending_regime = chop_aligned[i] < 38.2
        # Chop > 61.8 = ranging (exit positions)
        ranging_regime = chop_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Exit conditions
        if position != 0:
            # Exit if: opposing alignment OR chop shifts to ranging
            if position == 1:  # Long
                if bearish_alignment or ranging_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:  # Short
                if bullish_alignment or ranging_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with regime and volume filters
        if position == 0:
            # Long: bullish alignment AND trending regime AND volume confirmation
            long_condition = bullish_alignment and trending_regime and volume_confirm
            
            # Short: bearish alignment AND trending regime AND volume confirmation
            short_condition = bearish_alignment and trending_regime and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0