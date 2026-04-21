#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_RSI_Filter_With_Volume
Hypothesis: Use 1-day KAMA direction for trend, combined with RSI overbought/oversold and volume spike on 4h.
Long when KAMA rising, RSI < 30, and volume > 2x average. Short when KAMA falling, RSI > 70, and volume > 2x average.
Exit when RSI crosses back to neutral (40-60 range). Uses volume to confirm momentum and avoid false signals.
Designed for 4h to limit trade frequency (target: 20-50/year) and reduce fee drift. Works in bull markets by buying dips in uptrend,
and in bear markets by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA) on daily close
    def kama(close, period=10, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # For vectorized calculation, we compute ER per point
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if volatility[i-period:i].sum() != 0:
                er[i] = change[i-period] / volatility[i-period:i].sum()
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[:period] = close[:period].mean() if period > 0 else close[0]
        for i in range(period, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close_1d, period=10, fast=2, slow=30)
    kama_rising = np.zeros_like(kama_vals, dtype=bool)
    kama_falling = np.zeros_like(kama_vals, dtype=bool)
    kama_rising[1:] = kama_vals[1:] > kama_vals[:-1]
    kama_falling[1:] = kama_vals[1:] < kama_vals[:-1]
    
    # Align to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # RSI on 4h close
    if len(prices) < 14:
        return np.zeros(n)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after RSI warmup
        # Skip if indicators not ready
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: KAMA rising, RSI oversold (<30), volume spike
            if kama_rising_aligned[i] and rsi[i] < 30 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling, RSI overbought (>70), volume spike
            elif kama_falling_aligned[i] and rsi[i] > 70 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back above 40 (neutral)
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses back below 60 (neutral)
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Direction_RSI_Filter_With_Volume"
timeframe = "4h"
leverage = 1.0