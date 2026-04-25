#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from daily chart act as strong support/resistance on 12h timeframe.
Breakout through these levels with daily EMA34 trend alignment and volume confirmation captures
swing moves while minimizing trades. Works in bull/bear by following daily trend. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 12h timeframe
    prev_high_12h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_12h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_12h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels: H3/L3
    rng = prev_high_12h - prev_low_12h
    h3 = prev_close_12h + rng * 1.1 / 6.0
    l3 = prev_close_12h - rng * 1.1 / 6.0
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla (1d) + EMA34 (1d) + VolMA20
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_level = ema_34_1d_aligned[i]
        
        # Volume spike: current volume > 2.5 * 20-period average
        volume_spike = curr_volume > 2.5 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = curr_close > h3[i]  # Break above H3
        bearish_breakout = curr_close < l3[i]  # Break below L3
        
        # Exit conditions: reverse breakout or trend change
        if position != 0:
            if position == 1 and (curr_close < l3[i] or curr_close < ema_34_level):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (curr_close > h3[i] or curr_close > ema_34_level):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Breakout + trend + volume
        if position == 0:
            long_condition = bullish_breakout and (curr_close > ema_34_level) and volume_spike
            short_condition = bearish_breakout and (curr_close < ema_34_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0