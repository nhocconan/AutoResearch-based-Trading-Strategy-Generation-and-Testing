#!/usr/bin/env python3
name = "4h_24hBreakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 24-hour high/low breakout (6 periods of 4h)
    high_24h = pd.Series(high).rolling(window=6, min_periods=6).max().shift(1).values
    low_24h = pd.Series(low).rolling(window=6, min_periods=6).min().shift(1).values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(6, 20)  # Need 6 for 24h breakout, 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 24h high AND above 12h EMA50 + volume
            if close[i] > high_24h[i] and close[i] > ema_50_12h_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24h low AND below 12h EMA50 + volume
            elif close[i] < low_24h[i] and close[i] < ema_50_12h_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to 24h range or breaks in opposite direction
            if position == 1:
                if close[i] < low_24h[i] or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_24h[i] or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals