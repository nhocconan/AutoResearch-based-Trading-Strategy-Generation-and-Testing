#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter: > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian + above 1d EMA34 + volume
            if (close[i] > high_20[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short: Breakout below lower Donchian + below 1d EMA34 + volume
            elif (close[i] < low_20[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Close below lower Donchian or below 1d EMA34
            if close[i] < low_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Close above upper Donchian or above 1d EMA34
            if close[i] > high_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 4h Donchian breakout with 1d EMA34 trend filter and volume confirmation.
# Works in bull markets (breakouts above rising EMA) and bear markets (breakouts below falling EMA).
# Volume filter ensures breakouts have participation. Position size 0.30 balances risk/return.
# Target: 30-50 trades/year to minimize fee drag. Uses discrete levels to reduce churn.