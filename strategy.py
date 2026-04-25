#!/usr/bin/env python3
"""
6h_WeeklyCamarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: Weekly Camarilla H3/L3 breakout on 6h with 1d EMA34 trend filter and volume confirmation.
Uses weekly pivots for longer-term structure, reducing false breakouts. Works in bull markets (breakouts with trend) and bear markets (fades from extremes with volume).
Targets 12-30 trades/year (50-120 over 4 years) with discrete sizing (0.25) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for weekly Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla levels: H3/L3
    camarilla_h3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_l3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align weekly levels to 6h timeframe (completed 1w bar only)
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly Camarilla (1 bar), EMA34 (34), volume MA (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_1w_aligned[i]) or 
            np.isnan(camarilla_l3_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above weekly H3 + 1d uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_1w_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below weekly L3 + 1d downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_1w_aligned[i]) and \
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
            # Exit: price closes below weekly L3 OR 1d trend turns down
            if (close[i] < camarilla_l3_1w_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above weekly H3 OR 1d trend turns up
            if (close[i] > camarilla_h3_1w_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyCamarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0