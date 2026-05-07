#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_Regime_Filter
Hypothesis: 12h KAMA trend direction + volume spike + 1d volatility regime filter.
KAMA adapts to market conditions (trending/ranging) reducing whipsaw. Volume confirms conviction.
Volatility regime (using ATR ratio) avoids trading in low-volatility chop.
Target: 20-40 trades/year to minimize fee drag while capturing strong trends.
Works in bull via trend following, in bear via avoiding false signals during chop.
"""

name = "12h_KAMA_Trend_With_Volume_Regime_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (12-period) - adapts to market noise
    # ER = Efficiency Ratio, smoother in trend, faster in range
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper KAMA calculation
    dir = np.abs(np.diff(close, k=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder
    
    # Recompute properly
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(change)
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + change[i] - (change[i-9] if i >= 9 else 0)
    
    er = np.zeros_like(close)
    er[10:] = dir[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get 1d data for volatility regime filter (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)+14):
        if i == 14:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i-14]) / 14
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma_50 = np.full_like(atr_1d, np.nan)
    for i in range(50, len(atr_1d)):
        atr_ma_50[i] = np.mean(atr_1d[i-50:i])
    
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume spike: current volume > 1.8x 30-period average (to reduce frequency)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    vol_spike = volume > (1.8 * vol_ma_30)
    
    # KAMA trend: price above/below KAMA
    kama_trend_up = close > kama
    kama_trend_down = close < kama
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 50)  # Warmup for volume, ATR, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when volatility is elevated (ATR ratio > 0.8)
        # Avoids low-volatility chop where whipsaw occurs
        if atr_ratio_aligned[i] < 0.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + volume spike
            if kama_trend_up[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + volume spike
            elif kama_trend_down[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA (trend change)
            if not kama_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if not kama_trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals