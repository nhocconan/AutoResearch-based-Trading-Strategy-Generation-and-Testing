#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 4h with 4h EMA34 trend filter and volume confirmation.
Uses 4h trend to capture intermediate market direction, reducing false breakouts.
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) limits fee drag.
Designed to work in both bull and bear markets by aligning with 4h trend.
Target: 19-50 trades/year per symbol (75-200 total over 4 years).
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
    
    # Get 4h data for HTF trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    camarilla_h3 = close_4h_vals + 1.1 * (high_4h - low_4h) / 4
    camarilla_l3 = close_4h_vals - 1.1 * (high_4h - low_4h) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA34)
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above H3 + 4h uptrend + volume confirmation
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_4h_bullish and volume_confirm[i]
            
            # Short setup: price breaks below L3 + 4h downtrend + volume confirmation
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_4h_bearish and volume_confirm[i]
            
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
            # Exit: price touches L3 (stop) OR 4h trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 (stop) OR 4h trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0