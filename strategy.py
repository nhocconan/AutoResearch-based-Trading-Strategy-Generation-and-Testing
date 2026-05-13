#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_Volume_Confirmation
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h to capture adaptive trend direction, confirmed by volume spike (>1.5x 20-period average) and filtered by 1-day EMA50 trend. Go long when KAMA turns upward with volume confirmation and price above daily EMA50, short when KAMA turns downward with volume confirmation and price below daily EMA50. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. Designed for 4h timeframe to limit trades (target: 20-50/year) and avoid fee drag.
"""

name = "4h_KAMA_Trend_Filter_Volume_Confirmation"
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
    
    # Calculate KAMA on 4h close
    # KAMA parameters: ER period=10, fast EMA=2, slow EMA=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Recalculate volatility properly: sum of absolute changes over ER period
    er_window = 10
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(change)
    for i in range(len(volatility)):
        start = max(0, i - er_window + 1)
        volatility[i] = np.sum(change[start:i+1])
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility  # Efficiency Ratio
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: KAMA turning upward (current > previous) + volume spike + price above daily EMA50
            if kama[i] > kama[i-1] and vol_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning downward (current < previous) + volume spike + price below daily EMA50
            elif kama[i] < kama[i-1] and vol_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns downward or price breaks below daily EMA50
            if kama[i] < kama[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns upward or price breaks above daily EMA50
            if kama[i] > kama[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals