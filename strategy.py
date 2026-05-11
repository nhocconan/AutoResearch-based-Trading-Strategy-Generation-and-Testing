#!/usr/bin/env python3
name = "1d_1w_KAMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA on 1d close - adaptive trend strength indicator
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))  # 10-period net change
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)  # Efficiency ratio
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 1w KAMA to 1d timeframe
    kama_1w_series = pd.Series(kama)
    kama_1w = kama_1w_series.ewm(span=30, min_periods=30).mean().values  # longer period for weekly
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume filter: current volume > 1.5x 50-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(kama_1w_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA(1w) AND volume filter
            if close[i] > kama_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA(1w) AND volume filter
            elif close[i] < kama_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA(1w)
            if close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA(1w)
            if close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals