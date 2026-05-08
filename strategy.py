#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d volatility regime filter and volume confirmation
# KAMA adapts to market conditions - fast in trends, slow in ranges.
# Combined with 1d ATR-based volatility regime (low vol = range, high vol = trend) 
# and volume confirmation to filter false signals.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.

name = "4h_KAMA_VolRegime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility regime and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)
    
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d ATR-based volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1d = np.maximum(high_1d - low_1d,
                      np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                 np.abs(low_1d - np.roll(close_1d, 1))))
    tr1d[0] = high_1d[0] - low_1d[0]  # First bar
    
    atr1d = np.zeros_like(high_1d)
    atr1d[0] = tr1d[0]
    for i in range(1, len(tr1d)):
        atr1d[i] = (atr1d[i-1] * 13 + tr1d[i]) / 14  # Wilder smoothing
    
    # Volatility regime: high volatility = trending market
    atr_ma = np.zeros_like(atr1d)
    for i in range(20, len(atr1d)):
        atr_ma[i] = np.mean(atr1d[i-20:i])
    
    vol_regime = atr1d > (atr_ma * 1.2)  # High volatility regime
    vol_regime_4h = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # 1d volume confirmation
    vol_ma_1d = np.zeros_like(df_1d['volume'].values)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(df_1d['volume'].values[i-20:i])
    
    vol_conf = df_1d['volume'].values > (vol_ma_1d * 1.5)
    vol_conf_4h = align_htf_to_ltf(prices, df_1d, vol_conf)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or not available
        if (np.isnan(kama[i]) or np.isnan(vol_regime_4h[i]) or 
            np.isnan(vol_conf_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA, high volatility regime, volume confirmation
            if close[i] > kama[i] and vol_regime_4h[i] and vol_conf_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, high volatility regime, volume confirmation
            elif close[i] < kama[i] and vol_regime_4h[i] and vol_conf_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or volatility regime changes
            if close[i] < kama[i] or not vol_regime_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or volatility regime changes
            if close[i] > kama[i] or not vol_regime_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals