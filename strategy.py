#!/usr/bin/env python3
"""
6h_WeeklyPivot_HL_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h breakout above/below weekly pivot H/L levels with 1d EMA34 trend filter and volume confirmation.
Weekly pivots provide structural support/resistance that works in both bull and bear markets.
Uses discrete position sizing (0.25) to limit fee drag. Targets 15-25 trades/year.
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
    
    # Get weekly data for pivot points (H/L)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: H = (H+L+C)/3 + (H-L), L = (H+L+C)/3 - (H-L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    weekly_h = pivot_1w + (high_1w - low_1w)  # R1 equivalent
    weekly_l = pivot_1w - (high_1w - low_1w)  # S1 equivalent
    
    # Align weekly pivot to 6h (completed weekly bar only)
    weekly_h_aligned = align_htf_to_ltf(prices, df_1w, weekly_h)
    weekly_l_aligned = align_htf_to_ltf(prices, df_1w, weekly_l)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 periods per day on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly pivot (1 bar), EMA34 (34), volume MA (24)
    start_idx = max(1, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_h_aligned[i]) or 
            np.isnan(weekly_l_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above weekly H + 1d uptrend + volume spike
            long_setup = (close[i] > weekly_h_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below weekly L + 1d downtrend + volume spike
            short_setup = (close[i] < weekly_l_aligned[i]) and \
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
            # Exit: price closes below weekly L OR 1d trend turns down
            if (close[i] < weekly_l_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above weekly H OR 1d trend turns up
            if (close[i] > weekly_h_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_HL_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0