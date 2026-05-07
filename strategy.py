#!/usr/bin/env python3
name = "1h_4d_Donchian_Breakout_1dTrend_Volume"
timeframe = "1h"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1h Donchian channels (20-period)
    high_20h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1h volume filter: > 1.5x 24-period average
    vol_ma_24h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.5 * vol_ma_24h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20h[i]) or 
            np.isnan(low_20h[i]) or np.isnan(vol_ma_24h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20h high + above 1d EMA34 + volume
            if (close[i] > high_20h[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 20h low + below 1d EMA34 + volume
            elif (close[i] < low_20h[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price breaks below 20h low OR closes below 1d EMA34
            if close[i] < low_20h[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price breaks above 20h high OR closes above 1d EMA34
            if close[i] > high_20h[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian breakout with 1d EMA34 trend filter and volume confirmation.
# Donchian breakouts capture momentum bursts; 1d EMA34 ensures alignment with daily trend.
# Volume filter ensures institutional participation. Works in bull (breakouts above EMA) 
# and bear (breakouts below EMA). Target: 20-50 trades/year to minimize fee drag.
# Position size 0.20 limits risk during 2022-like drawdowns. 1h timeframe balances 
# responsiveness with noise reduction vs lower timeframes.