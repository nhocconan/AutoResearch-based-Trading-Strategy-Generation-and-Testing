#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 4h with 1d ATR-based trend filter and volume confirmation.
H3/L3 levels offer stronger breakout signals than R1/S1. Trend filter uses 1d ATR position relative to 1d SMA50
to capture momentum in both bull and bear markets. Volume spike confirms institutional participation.
Designed for lower trade frequency (<50/year) to minimize fee drag while maintaining edge in ranging and trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend context
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility-based trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - close_1d_shift))
    tr2 = np.maximum(np.abs(low_1d - close_1d_shift), np.abs(close_1d - close_1d_shift))
    tr = np.maximum(tr1, tr2)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: price > SMA50 + 0.5*ATR = bullish, price < SMA50 - 0.5*ATR = bearish
    # This creates a deadzone around SMA50 to reduce whipsaw
    bullish_threshold = sma_50_1d + (0.5 * atr_14_1d)
    bearish_threshold = sma_50_1d - (0.5 * atr_14_1d)
    htf_1d_bullish = close_1d > bullish_threshold
    htf_1d_bearish = close_1d < bearish_threshold
    
    # Align HTF arrays to 4h timeframe
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    bullish_threshold_aligned = align_htf_to_ltf(prices, df_1d, bullish_threshold)
    bearish_threshold_aligned = align_htf_to_ltf(prices, df_1d, bearish_threshold)
    htf_1d_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_1d_bullish.astype(float))
    htf_1d_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_1d_bearish.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_h3_1d = c_1d + (range_1d * 1.1 / 4.0)   # H3 = C + (range * 1.1/4)
    camarilla_l3_1d = c_1d - (range_1d * 1.1 / 4.0)   # L3 = C - (range * 1.1/4)
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: 4h volume > 1.8 * 20-period average (slightly looser for more signals)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for SMA50 (50), ATR (14+14=28), volume MA (20)
    start_idx = max(50, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(bullish_threshold_aligned[i]) or np.isnan(bearish_threshold_aligned[i]) or
            np.isnan(htf_1d_bullish_aligned[i]) or np.isnan(htf_1d_bearish_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend with deadzone
        htf_1d_bullish = bool(htf_1d_bullish_aligned[i])
        htf_1d_bearish = bool(htf_1d_bearish_aligned[i])
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + 1d bullish trend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below Camarilla L3 + 1d bearish trend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla L3 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0