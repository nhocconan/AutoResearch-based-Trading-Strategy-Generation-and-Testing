#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for volume and chop regime.
- Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift).
- In trending markets (CHOP < 38.2): Go long when Lips cross above Teeth and Teeth above Jaw (bullish alignment).
                           Go short when Lips cross below Teeth and Teeth below Jaw (bearish alignment).
- In ranging markets (CHOP > 61.8): Fade extremes - long when price touches lower band and reverses up,
                                   short when price touches upper band and reverses down.
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA to avoid false signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
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
    
    # Get 1d data for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ma_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, tr_sum_14 / range_14, 1.0)
    chop_raw = np.maximum(chop_raw, 1e-10)  # Avoid log(0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1d indicators to 12h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Williams Alligator on 12h
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (shift right = add NaN at beginning)
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13+8, 8+5, 5+3)  # Need enough for volume MA, SMMA calculations, and shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume_spike_aligned[i]
        chop_val = chop_aligned[i]
        curr_close = close[i]
        curr_low = low[i]
        curr_high = high[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals only with volume confirmation
            if vol_spike:
                if chop_val < 38.2:  # Trending regime: Alligator alignment
                    # Bullish: Lips > Teeth > Jaw
                    if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish: Lips < Teeth < Jaw
                    elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: fade extremes
                    # Calculate 12h Bollinger-like bands for entry zones
                    # Using 20-period SMA and 2 standard deviations
                    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
                    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
                    upper_band = sma_20 + (2.0 * std_20)
                    lower_band = sma_20 - (2.0 * std_20)
                    
                    # Long when price touches lower band and shows reversal (close > low)
                    if not np.isnan(lower_band[i]) and curr_low <= lower_band[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper band and shows reversal (close < high)
                    elif not np.isnan(upper_band[i]) and curr_high >= upper_band[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if chop_val < 38.2:  # Trending: exit on Alligator reversal
                if lips[i] < teeth[i]:  # Lips cross below Teeth
                    exit_signal = True
            else:  # Ranging: exit when price reaches middle or opposite band
                sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
                std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
                upper_band = sma_20 + (2.0 * std_20)
                lower_band = sma_20 - (2.0 * std_20)
                mid_band = sma_20
                if not np.isnan(mid_band[i]) and curr_close >= mid_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if chop_val < 38.2:  # Trending: exit on Alligator reversal
                if lips[i] > teeth[i]:  # Lips cross above Teeth
                    exit_signal = True
            else:  # Ranging: exit when price reaches middle or opposite band
                sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
                std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
                upper_band = sma_20 + (2.0 * std_20)
                lower_band = sma_20 - (2.0 * std_20)
                mid_band = sma_20
                if not np.isnan(mid_band[i]) and curr_close <= mid_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeChop_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0