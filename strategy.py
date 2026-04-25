#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1wTrend_1dVolumeSpike
Hypothesis: Camarilla R1/S1 levels from 1d act as intraday support/resistance. A break above R1 with 1d volume spike and 1w uptrend signals long; break below S1 with 1d volume spike and 1w downtrend signals short. Uses discrete position sizing (0.25) to limit fee drag. Works in both bull and bear markets by aligning with 1w trend while using 1d structure for precise entries. Target: 12-37 trades/year per symbol.
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
    
    # 1d data for Camarilla calculation and volume spike (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from 1d OHLC (previous day's values)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous day's close to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's OHLC (avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # first value has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    
    # Align HTF indicators to LTF (6h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Camarilla calculation (requires previous day data)
    start_idx = 1  # because we rolled by 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with 1d volume spike and 1w uptrend
            long_breakout = (curr_close > r1_aligned[i]) and vol_spike_1d_aligned[i] and (curr_close > ema_50_1w_aligned[i])
            # Short: price breaks below S1 with 1d volume spike and 1w downtrend
            short_breakout = (curr_close < s1_aligned[i]) and vol_spike_1d_aligned[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if (curr_close < s1_aligned[i]) or (curr_close < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if (curr_close > r1_aligned[i]) or (curr_close > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1wTrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0