#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator + 1-week trend filter (EMA50) + volume confirmation.
Williams Alligator identifies convergence/divergence of smoothed medians (Jaws/Teeth/Lips).
Long when Lips > Teeth > Jaws (bullish alignment), Short when Lips < Teeth < Jaws (bearish).
Weekly EMA50 filter ensures alignment with higher-timeframe trend.
Volume > 1.5x average confirms momentum.
Designed for low-frequency signals (<25/year) to avoid fee drag, works in bull/bear by
capturing trending moves aligned with weekly structure.
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
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaws: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (similar to Wilder smoothing)
    
    median_price = (high + low) / 2
    median_1d = median_price
    
    # SMMA calculation (Wilder smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws_raw = smma(median_1d, 13)
    teeth_raw = smma(median_1d, 8)
    lips_raw = smma(median_1d, 5)
    
    # Apply shifts (Jaws +8, Teeth +5, Lips +3)
    jaws = np.full_like(jaws_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaws_raw) > 8:
        jaws[8:] = jaws_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on weekly
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align indicators to 1d timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Alligator (13+8=21), EMA50 (50), volume MA (20)
    start_idx = max(21, 50, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
        
        # Weekly trend filter
        price = close[i]
        above_weekly_ema = price > ema_50_aligned[i]
        below_weekly_ema = price < ema_50_aligned[i]
        
        # Volume confirmation
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: bullish alignment + above weekly EMA + volume
            if bullish_alignment and above_weekly_ema and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish alignment + below weekly EMA + volume
            elif bearish_alignment and below_weekly_ema and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: alignment breaks or price crosses below weekly EMA
            if not bullish_alignment or price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: alignment breaks or price crosses above weekly EMA
            if not bearish_alignment or price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsAlligator_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0