#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_with_Volume_and_Chop
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with volume confirmation and Choppiness Index regime filter to avoid whipsaws.
KAMA adapts to market noise, reducing false signals in sideways markets. Volume confirms
strength of breakouts. Choppiness Index > 61.8 indicates ranging (avoid trend trades),
< 38.2 indicates trending (favor trend trades). Designed for low trade frequency
(<25/year) to minimize fee drag on 1d timeframe. Works in bull/bear by adapting to
regime: uses KAMA for trend, volume for confirmation, chop for regime filter.
"""

name = "1d_KAMA_Trend_Filter_with_Volume_and_Chop"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) Calculation ===
    # Parameters: ER period=10, Fast SC=2, Slow SC=30
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=er_period))  # |close - close[er_period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over er_period
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing Constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Choppiness Index Calculation ===
    # Parameters: period=14
    chop_period = 14
    atr = np.zeros_like(close)
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    # True Range average
    atr_sum = np.zeros_like(close)
    for i in range(chop_period, len(close)):
        atr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    atr = atr_sum / chop_period
    # Max/min high-low over period
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    # Chop = 100 * log10(sum(atr) / (max_high - min_low)) / log10(period)
    range_val = max_high - min_low
    chop = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        if range_val[i] > 0 and atr[i] > 0:
            chop[i] = 100 * np.log10(atr[i] * chop_period / range_val[i]) / np.log10(chop_period)
        else:
            chop[i] = 50  # neutral
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x volume average
    
    # === Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA, chop, volume calculations)
    start_idx = max(er_period, chop_period, 20) + 10
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA AND trending regime (Chop < 38.2) AND volume spike
            if (close[i] > kama[i] and 
                chop[i] < 38.2 and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA AND trending regime (Chop < 38.2) AND volume spike
            elif (close[i] < kama[i] and 
                  chop[i] < 38.2 and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses back through KAMA OR chop indicates ranging (Chop > 61.8)
            if position == 1:
                if close[i] < kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > kama[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals