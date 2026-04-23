#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-period high AND close > 1d EMA50 (uptrend) AND volume > 2.0x 20-period MA.
Short when price breaks below 20-period low AND close < 1d EMA50 (downtrend) AND volume > 2.0x 20-period MA.
Exit when price returns to midpoint of the Donchian channel or opposite breakout occurs.
Designed for ~12-25 trades/year with structure-based edge, avoiding overtrading.
Donchian channels provide clear breakout levels; 1d EMA50 ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian(20) channels
    donchian_period = 20
    high_ma = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_ma = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (high_ma + low_ma) / 2.0
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_period, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_ma[i]  # Break above 20-period high
        breakout_down = close[i] < low_ma[i]  # Break below 20-period low
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.1 * (high_ma[i] - low_ma[i])  # Near midpoint
        opposite_breakout = (position == 1 and breakout_down) or \
                            (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above high AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below low AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to midpoint or opposite breakout
            exit_signal = False
            if position == 1:
                exit_signal = return_to_mid or opposite_breakout
            elif position == -1:
                exit_signal = return_to_mid or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0