#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Range_200MA_v1
Hypothesis: Uses KAMA for trend direction on 1d timeframe, combined with RSI for mean-reversion entries
and 200-day moving average filter to avoid counter-trend trades. Designed for low trade frequency
to work in both bull and bear markets by adapting to trend conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA, RSI, and MA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 200-day moving average
    ma200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align indicators to 1d timeframe (same as primary)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ma200_aligned = align_htf_to_ltf(prices, df_1d, ma200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 200  # MA200 needs 200 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ma200_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        ma200_val = ma200_aligned[i]
        
        if position == 0:
            # Determine trend: price vs MA200
            uptrend = close_val > ma200_val
            downtrend = close_val < ma200_val
            
            # Long conditions: uptrend + RSI oversold
            if uptrend and rsi_val < 30:
                signals[i] = size
                position = 1
            # Short conditions: downtrend + RSI overbought
            elif downtrend and rsi_val > 70:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: RSI overbought or trend change
            if rsi_val > 70 or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: RSI oversold or trend change
            if rsi_val < 30 or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_Range_200MA_v1"
timeframe = "1d"
leverage = 1.0