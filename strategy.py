#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR-based regime detection (choppy vs trending) and volume spike filter.
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) of median price.
- Regime: ATR(10)/ATR(30) ratio > 1.2 = trending (favor Alligator alignment), < 0.8 = choppy (avoid trading).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND trending regime AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND trending regime AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips crosses Teeth) OR regime shifts to choppy (ATR ratio < 0.8).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading strong trends, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's MA or RMA"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Calculate 1d ATR(10) and ATR(30) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR30
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
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for regime: >1.2 = trending, <0.8 = choppy
    atr_ratio = atr10 / atr30
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price
    jaw = smma(median_price, 13)
    # Teeth: 8-period SMMA of median price
    teeth = smma(median_price, 8)
    # Lips: 5-period SMMA of median price
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30)  # Need 13 for Jaw, 30 for ATR30
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade in trending markets (ATR ratio > 1.2)
        trending_regime = atr_ratio_aligned[i] > 1.2
        # Choppy regime filter: exit if ATR ratio < 0.8
        choppy_regime = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Alligator alignment conditions
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Exit conditions
        if position != 0:
            # Exit if regime turns choppy
            if choppy_regime:
                signals[i] = 0.0
                position = 0
                continue
            # Exit long: bullish alignment breaks
            elif position == 1:
                if not bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bearish alignment breaks
            elif position == -1:
                if not bearish_alignment:
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

name = "12h_WilliamsAlligator_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0