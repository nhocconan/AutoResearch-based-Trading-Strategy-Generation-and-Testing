#!/usr/bin/env python3
"""
12h_PivotBreakout_1dTrend_Volume
Hypothesis: Trading breakouts from 12-hour price channels confirmed by daily trend (EMA50) and volume spikes.
In bull markets, price tends to break above channel resistance with strong volume; in bear markets,
it breaks below support. The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend,
reducing false breakouts in ranging markets. Volume confirmation filters weak breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_PivotBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # 12-period high/low for price channel (using 12h data)
    high_12 = np.full(n, np.nan)
    low_12 = np.full(n, np.nan)
    if n >= 12:
        # Initialize first value
        high_12[11] = np.max(high[:12])
        low_12[11] = np.min(low[:12])
        # Rolling max/min
        for i in range(12, n):
            high_12[i] = max(high_12[i-1], high[i])
            low_12[i] = min(low_12[i-1], low[i])
            # Remove values that fall out of the 12-period window
            if i >= 24:
                if high_12[i-12] == high_12[i-1]:
                    high_12[i] = np.max(high[i-11:i+1])
                if low_12[i-12] == low_12[i-1]:
                    low_12[i] = np.min(low[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12, 50)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(high_12[i]) or np.isnan(low_12[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume approximation: 12h volume from daily (12h = 1/2 day)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: price breaks above 12-period high with uptrend and volume
            if close[i] > high_12[i] and close[i] > ema50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12-period low with downtrend and volume
            elif close[i] < low_12[i] and close[i] < ema50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below channel low or trend reversal
            if close[i] < low_12[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above channel high or trend reversal
            if close[i] > high_12[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals