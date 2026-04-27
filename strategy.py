#!/usr/bin/env python3
"""
Hypothesis: 4h strategy combining daily Williams Alligator (Jaw/Teeth/Lips) with EMA50 trend filter and volume confirmation.
The Alligator identifies trends via convergence/divergence of smoothed medians. EMA50 confirms trend direction.
Volume > 1.5x average confirms momentum. Works in bull/bear by capturing sustained trends with clear exit conditions.
Target: 20-40 trades/year to minimize fee drag.
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2
    
    def smoothed_median(median, length):
        """Williams Alligator uses SMMA (Smoothed Moving Average)"""
        smoothed = np.full_like(median, np.nan)
        if len(median) < length:
            return smoothed
        smoothed[length-1] = np.mean(median[:length])
        for i in range(length, len(median)):
            smoothed[i] = (smoothed[i-1] * (length - 1) + median[i]) / length
        return smoothed
    
    # Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw = smoothed_median(median_1d, 13)
    teeth = smoothed_median(median_1d, 8)
    lips = smoothed_median(median_1d, 5)
    
    # Apply shifts (Williams Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set initial values to NaN due to shift
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe (waits for 1d bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # EMA50 trend filter on 1d
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * multiplier) + (ema50_1d[i-1] * (1 - multiplier))
    
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation on 4h
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Alligator (13+8), EMA50 (50), volume MA (20)
    start_idx = max(13+8, 50, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        # Alligator signals: Lips above Teeth above Jaw = bullish alignment
        # Lips below Teeth below Jaw = bearish alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: bullish Alligator alignment + bullish trend + volume
            if bullish_alignment and bullish_trend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish Alligator alignment + bearish trend + volume
            elif bearish_alignment and bearish_trend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator convergence (Lips crosses below Teeth) or trend change
            if lips_aligned[i] < teeth_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Alligator convergence (Lips crosses above Teeth) or trend change
            if lips_aligned[i] > teeth_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0