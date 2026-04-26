#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_Filter
Hypothesis: On 6h timeframe, combine ADX (>25) with Williams Alligator (Jaw/Teeth/Lips alignment) to identify strong trends. Enter long when price > Lips AND Lips > Teeth AND Jaw (bullish alignment) AND ADX rising. Enter short when price < Lips AND Lips < Teeth AND Jaw (bearish alignment) AND ADX rising. Exit when ADX falls below 20 (trend weakening) or Alligator lines cross (trend reversal). Uses 1d HTF for ADX calculation to reduce noise and avoid whipsaw in ranging markets. Targets 12-30 trades/year on BTC/ETH/SOL with controlled fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX and Alligator calculation (less noisy on higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / np.maximum(tr_smoothed, 1e-10)
    minus_di = 100 * minus_dm_smoothed / np.maximum(tr_smoothed, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # Calculate 1d Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Shift the lines (Jaw 8, Teeth 5, Lips 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First values become NaN due to roll
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align all indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX and Alligator warmup
    start_idx = max(34, 18)  # ADX needs ~34, Alligator lips needs 5+3=8
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Alligator alignment conditions
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # ADX conditions
        adx_strong = adx_aligned[i] > 25.0
        adx_weak = adx_aligned[i] < 20.0
        
        # Price vs Lips
        price_above_lips = close[i] > lips_aligned[i]
        price_below_lips = close[i] < lips_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + price above lips + strong ADX
            long_signal = bullish_alignment and price_above_lips and adx_strong
            
            # Short: bearish alignment + price below lips + strong ADX
            short_signal = bearish_alignment and price_below_lips and adx_strong
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX weak OR bearish alignment OR price below lips
            if adx_weak or bearish_alignment or price_below_lips:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX weak OR bullish alignment OR price above lips
            if adx_weak or bullish_alignment or price_above_lips:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0