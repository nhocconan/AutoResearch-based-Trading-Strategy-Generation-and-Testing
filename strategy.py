#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Trade Camarilla H3/L3 breakouts on 12h with 1w EMA50 trend filter and volume confirmation. Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol. Works in bull/bear via 1w trend alignment - in uptrend only long H3 breakouts, in downtrend only short L3 breakdowns. Volume spike confirms institutional participation. Designed for lower trade frequency on 12h timeframe to minimize fee impact while capturing significant moves.
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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 (using close of previous 1d bar)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 30-period volume MA (stricter for 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (30)
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
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

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0