#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 12h with 1w EMA50 trend filter and volume confirmation.
Camarilla levels provide high-probability reversal/breakout points; 1w trend ensures alignment with major market direction.
Volume confirmation filters false breakouts. Designed for low turnover (12-37 trades/year) to minimize fee drag.
Works in both bull and bear markets by following 1w trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 are key breakout levels
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed as they're based on completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirm = volume > (vol_ma * 1.5)
    else:
        volume_confirm = np.ones(n, dtype=bool)  # No volume filter if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and Camarilla (need at least 1 1d bar)
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above H3 + 1w uptrend + volume confirmation
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1w_bullish and volume_confirm[i]
            
            # Short setup: price breaks below L3 + 1w downtrend + volume confirmation
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1w_bearish and volume_confirm[i]
            
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
            # Exit: price touches L3 (stop) OR 1w trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 (stop) OR 1w trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0