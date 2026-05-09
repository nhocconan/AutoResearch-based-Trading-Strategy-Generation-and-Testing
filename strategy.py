#!/usr/bin/env python3
# 6h_KAMA_Trend_Filter
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) as adaptive trend filter on 6h timeframe.
# KAMA adapts to market conditions - slows in ranging markets, speeds in trending markets.
# Combined with 12h EMA50 for higher timeframe trend confirmation and volume spike for confirmation.
# Works in bull/bear: Trend filter avoids counter-trend trades, volume confirms institutional interest.
# Uses adaptive smoothing to reduce whipsaw in ranging markets while capturing trends.

name = "6h_KAMA_Trend_Filter"
timeframe = "6h"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 6h
    # KAMA parameters: ER (Efficiency Ratio) period 10, Fastest EMA 2, Slowest EMA 30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate change and volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Initialize arrays
    er = np.zeros(n)
    sc = np.zeros(n)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    for i in range(er_period, n):
        # Directional change over er_period
        directional_change = np.abs(close[i] - close[i-er_period])
        # Sum of absolute changes over er_period
        total_change = np.sum(volatility[i-er_period+1:i+1])
        if total_change > 0:
            er[i] = directional_change / total_change
        else:
            er[i] = 0
        
        # Smoothing constant
        sc[i] = (er[i] * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        if i >= er_period:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # Calculate 12h EMA50 for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND above 12h EMA50 AND volume spike
            if (close[i] > kama[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND below 12h EMA50 AND volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR below 12h EMA50
            if close[i] < kama[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR above 12h EMA50
            if close[i] > kama[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals