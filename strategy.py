#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_With_Entry_Timing
Hypothesis: Use 1d KAMA trend for direction, 4h price action for entry timing.
- Long when: 1d KAMA rising AND price crosses above 4h SMA20 (pullback buy in uptrend)
- Short when: 1d KAMA falling AND price crosses below 4h SMA20 (sell the rally in downtrend)
- Exit when: price crosses back to opposite side of 4h SMA20
Uses volume filter to avoid low-liquidity entries. Targets 20-40 trades/year.
"""

name = "4h_1d_KAMA_Trend_With_Entry_Timing"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: KAMA ---
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.sum(volatility[np.arange(1, len(close_1d))[:, None] <= np.arange(1, len(close_1d))], axis=1) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = kama_1d[0]
    kama_rising = kama_1d > kama_1d_prev
    kama_falling = kama_1d < kama_1d_prev
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # --- 4h SMA20 for entry timing ---
    sma20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    
    # --- Volume filter: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # for KAMA and SMA20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma20_4h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d KAMA trend with volume
            if close_4h[i] > sma20_4h[i] and kama_rising_aligned[i] and vol_ok:
                # Long: price above SMA20 in uptrend
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < sma20_4h[i] and kama_falling_aligned[i] and vol_ok:
                # Short: price below SMA20 in downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses back to opposite side of SMA20
            if position == 1:
                if close_4h[i] < sma20_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_4h[i] > sma20_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals