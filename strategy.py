#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Camarilla H3/L3 breakouts with weekly trend filter (price > weekly EMA34) and volume confirmation (>2.0x 20-bar MA). 
Camarilla H3/L3 are strong breakout levels that reduce false signals. Weekly EMA34 provides a smooth trend filter to avoid counter-trend trades in bear markets. 
Volume confirmation ensures breakouts have conviction. Discrete sizing 0.25 balances profit and fee drag. 
Target: 15-25 trades/year (~60-100 over 4 years) to stay within fee drag limits for 1d timeframe.
Works in both bull and bear: weekly trend filter prevents shorts in strong uptrends and longs in strong downtrends, while volume confirmation filters weak breakouts.
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 1d timeframe (no alignment needed as already 1d)
    h3_aligned = h3  # Already aligned to 1d bars
    l3_aligned = l3  # Already aligned to 1d bars
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter for lower trade frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND weekly trend bullish (close > weekly EMA34) AND volume confirm
            long_setup = (close[i] > h3_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below L3 AND weekly trend bearish (close < weekly EMA34) AND volume confirm
            short_setup = (close[i] < l3_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Camarilla H3/L3 range OR weekly trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR weekly trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0