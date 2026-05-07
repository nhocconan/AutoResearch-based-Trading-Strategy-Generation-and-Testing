#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for Donchian and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + uptrend + volume confirmation
            if close[i] > high_max[i] and close[i] > ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + downtrend + volume confirmation
            elif close[i] < low_min[i] and close[i] < ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below lower Donchian or trend reversal
            if close[i] < low_min[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above upper Donchian or trend reversal
            if close[i] > high_max[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-period high, is above 1d EMA34 (uptrend), and volume confirms.
# Short when price breaks below 20-period low, is below 1d EMA34 (downtrend), and volume confirms.
# Uses 1d EMA for trend to avoid whipsaws, 4h Donchian for breakout signals.
# Volume filter (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (breakouts + uptrend) and bear markets (breakdowns + downtrend).
# Target: 20-50 trades/year to minimize fee drag while capturing sustained moves.