# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Williams_Alligator_Triple_Signal_V1
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 12h timeframe provides clear trend direction.
Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
Uses 1-week trend filter (close > EMA50 for long, close < EMA50 for short) to avoid counter-trend trades.
Volume confirmation (volume > 1.5x 24-period average) reduces false signals.
Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
Works in bull/bear by following trend with strong filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_period = 13
    jaw_offset = 8
    teeth_period = 8
    teeth_offset = 5
    lips_period = 5
    lips_offset = 3
    
    jaw_raw = smma((high_12h + low_12h) / 2, jaw_period)
    teeth_raw = smma((high_12h + low_12h) / 2, teeth_period)
    lips_raw = smma((high_12h + low_12h) / 2, lips_period)
    
    # Apply offsets (shift right by offset periods)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > jaw_offset:
        jaw[jaw_offset:] = jaw_raw[:-jaw_offset]
    if len(teeth_raw) > teeth_offset:
        teeth[teeth_offset:] = teeth_raw[:-teeth_offset]
    if len(lips_raw) > lips_offset:
        lips[lips_offset:] = lips_raw[:-lips_offset]
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align 1w EMA50 to lower timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready
    start_idx = max(jaw_period + jaw_offset, teeth_period + teeth_offset, lips_period + lips_offset, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        lips_gt_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_gt_jaw = teeth_aligned[i] > jaw_aligned[i]
        bullish_alignment = lips_gt_teeth and teeth_gt_jaw
        
        lips_lt_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_lt_jaw = teeth_aligned[i] < jaw_aligned[i]
        bearish_alignment = lips_lt_teeth and teeth_lt_jaw
        
        # 1-week trend filter
        uptrend_filter = close[i] > ema_1w_aligned[i]
        downtrend_filter = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish alignment + uptrend filter + volume
            if bullish_alignment and uptrend_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + downtrend filter + volume
            elif bearish_alignment and downtrend_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or price below 1w EMA50
            if bearish_alignment or not uptrend_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or price above 1w EMA50
            if bullish_alignment or not downtrend_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Triple_Signal_V1"
timeframe = "12h"
leverage = 1.0