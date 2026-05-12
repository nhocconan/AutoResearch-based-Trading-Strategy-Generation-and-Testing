#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter_Donchian_Breakout_With_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high and KAMA > KAMA.prev (uptrend) with volume > 1.5x average.
# Enter short when price breaks below Donchian(20) low and KAMA < KAMA.prev (downtrend) with volume > 1.5x average.
# Exit when price crosses back inside Donchian channel (mean reversion).
# Uses KAMA for adaptive trend, Donchian for breakout structure, volume for confirmation.
# Designed to work in both bull (breakouts) and bear (breakdowns) with low trade frequency.

name = "4h_KAMA_Trend_Filter_Donchian_Breakout_With_Volume"
timeframe = "4h"
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
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0, keepdims=True)
    # Fix volatility calculation: rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_len, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high with KAMA up and volume confirmation
            if close[i] > donch_high and kama_val > kama_prev and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with KAMA down and volume confirmation
            elif close[i] < donch_low and kama_val < kama_prev and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside Donchian channel (mean reversion)
            if close[i] < donch_high and close[i] > donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside Donchian channel (mean reversion)
            if close[i] < donch_high and close[i] > donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals