#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_v1
Hypothesis: Use 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
Long when price breaks above 20-period high and closes above 1d EMA50 with volume > 1.5x average.
Short when price breaks below 20-period low and closes below 1d EMA50 with volume > 1.5x average.
Fixed position size 0.25. ATR(20) stoploss to limit drawdown.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear via trend filter and volume confirmation.
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(20) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, close above EMA50, volume confirmation
            if high[i] > high_max[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, close below EMA50, volume confirmation
            elif low[i] < low_min[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA50 or ATR-based stoploss
            if close[i] < ema_50_aligned[i] or close[i] < high[i] - 2.0 * atr[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA50 or ATR-based stoploss
            if close[i] > ema_50_aligned[i] or close[i] > low[i] + 2.0 * atr[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0