#!/usr/bin/env python3
# 1D_KAMA_1WTrend_Volume
# Hypothesis: 1d strategy using KAMA direction for trend, weekly EMA34 filter, and volume confirmation.
# Enters long when KAMA direction is up, close > weekly EMA34, and volume > 1.5x average.
# Enters short when KAMA direction is down, close < weekly EMA34, and volume > 1.5x average.
# Exits when KAMA direction reverses. Designed to avoid overtrading with strict entry conditions.
# Uses weekly trend filter to work in both bull and bear markets by only trading in direction of higher timeframe trend.
# Target: 1d timeframe with weekly HTF for trend filter.

name = "1D_KAMA_1WTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) direction
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, n=10))  # direction over 10 periods
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    er = np.where(volatility > 0, dir / volatility, 0)
    # Smooth ER with smoothing constants (fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645)
    sc = (er * 0.605 + 0.0645) ** 2  # SC = [ER*(fastest-slowest) + slowest]^2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(kama > np.roll(kama, 1), 1, np.where(kama < np.roll(kama, 1), -1, 0))
    
    # Calculate EMA34 for trend filter (weekly)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection: 1.5x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 10)  # Ensure we have volume MA, EMA34, and KAMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or 
            np.isnan(kama_dir[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA direction up, close > weekly EMA34 (uptrend), volume spike (>1.5x)
            if (kama_dir[i] == 1 and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA direction down, close < weekly EMA34 (downtrend), volume spike (>1.5x)
            elif (kama_dir[i] == -1 and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA direction turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA direction turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals