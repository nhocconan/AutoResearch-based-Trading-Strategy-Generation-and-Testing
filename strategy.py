#!/usr/bin/env python3
name = "1d_WeeklyKeltnerBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA20 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_ltf_to_htf(prices, df_1w, ema_20_1w)
    
    # Daily Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper = ema_20 + 2.0 * atr
    lower = ema_20 - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close above weekly EMA20 and upper Keltner band
            if close[i] > ema_20_1w_aligned[i] and close[i] > upper[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly EMA20 and lower Keltner band
            elif close[i] < ema_20_1w_aligned[i] and close[i] < lower[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below EMA20
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above EMA20
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals