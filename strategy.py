#!/usr/bin/env python3
"""
6h_WeeklyCamarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade weekly Camarilla H3/L3 breakouts on 6h timeframe only when 1d EMA34 confirms trend and volume spikes (>2.0x 20-bar MA). Weekly Camarilla provides stronger institutional levels than daily, reducing false breakouts. Trend filter avoids counter-trend trades in choppy markets. Volume spike confirms institutional participation. Designed for low trade frequency (12-30/year) to minimize fee drag while capturing significant moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla levels from previous week
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_range = (high_1w - low_1w) * 1.1 / 4.0
    camarilla_H3 = close_1w + camarilla_range
    camarilla_L3 = close_1w - camarilla_range
    
    # Align Weekly Camarilla levels to 6h (completed weekly bar only)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h (completed daily bar only)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Weekly Camarilla (1w), EMA34 (1d), volume MA (20)
    start_idx = max(20, 34)  # 34 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Weekly Camarilla H3 + above 1d EMA34 + volume spike
            long_setup = (close[i] > camarilla_H3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Weekly Camarilla L3 + below 1d EMA34 + volume spike
            short_setup = (close[i] < camarilla_L3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
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
            # Exit: price closes below Weekly Camarilla L3 OR below 1d EMA34
            if (close[i] < camarilla_L3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Weekly Camarilla H3 OR above 1d EMA34
            if (close[i] > camarilla_H3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyCamarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0