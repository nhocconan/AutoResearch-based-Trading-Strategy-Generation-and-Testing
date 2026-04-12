#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_v1
Hypothesis: 4-hour Donchian channel breakout with volume confirmation and 1d trend filter.
Enters long on breakout above 20-period high with volume spike and 1d uptrend; short on breakdown below 20-period low with volume spike and 1d downtrend.
Uses fixed position sizing (0.25) to limit trades and reduce fee drag. Designed to capture strong trending moves while avoiding chop.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag while capturing strong moves.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period Donchian channels on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volume filter threshold
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average (avoid low-volume breakouts)
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Trend filter from 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        long_breakout = close[i] > high_20[i] and volume_filter and uptrend_1d
        short_breakout = close[i] < low_20[i] and volume_filter and downtrend_1d
        
        # Exit conditions: opposite Donchian breakout
        long_exit = close[i] < low_20[i]
        short_exit = close[i] > high_20[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0