#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation (>2x 24-bar avg). Enters long when price breaks above H3 in 1d uptrend, short when breaks below L3 in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~12-37 trades/year, works in bull/bear by following 1d trend filter.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 24-period average (24*12h = 12d)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 24-period data for volume MA and 50 for 1d EMA
    start_idx = max(24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Get previous completed 1d bar for Camarilla calculation
        idx_1d = i // 2  # 2*12h bars = 1d bar (approximate, but aligned via HTF data)
        if idx_1d < 1:
            signals[i] = 0.0
            continue
            
        prev_close_1d = close_1d[idx_1d - 1]
        prev_high_1d = high_1d[idx_1d - 1]
        prev_low_1d = low_1d[idx_1d - 1]
        
        # Camarilla levels calculation (H3/L3)
        range_1d = prev_high_1d - prev_low_1d
        camarilla_h3 = prev_close_1d + (range_1d * 1.1 / 4)
        camarilla_l3 = prev_close_1d - (range_1d * 1.1 / 4)
        
        if position == 0:
            # Long: price breaks above H3 in 1d uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_h3) and \
                              (close_1d[idx_1d] > ema_50_1d_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below L3 in 1d downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_l3) and \
                              (close_1d[idx_1d] < ema_50_1d_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR trend turns down
            if (curr_close < camarilla_l3) or \
               (close_1d[idx_1d] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR trend turns up
            if (curr_close > camarilla_h3) or \
               (close_1d[idx_1d] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0