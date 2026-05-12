#!/usr/bin/env python3
# 4h_KAMA_Trend_12hVolatilityFilter_Volume
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction,
# filtered by 12h volatility regime (low ATR ratio = trend-favorable),
# with volume confirmation (>1.5x 20-period average) to ensure institutional participation.
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
# Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.
# Works in bull/bear markets by following adaptive trend with volatility filter to avoid chop.

name = "4h_KAMA_Trend_12hVolatilityFilter_Volume"
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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate KAMA on 4h close
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.full_like(close, np.nan, dtype=float)
        kama_out[er_len] = close[er_len]
        for i in range(er_len+1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close)
    
    # 12h data for volatility filter (ATR ratio)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(14) on 12h
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period average ATR
    atr_ma_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_12h / atr_ma_50
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama_val)  # same timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR ratio < 1.0 (below average volatility)
        vol_filter = atr_ratio_aligned[i] < 1.0
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + low volatility regime
            if (close[i] > kama_aligned[i] and 
                volume_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume spike + low volatility regime
            elif (close[i] < kama_aligned[i] and 
                  volume_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals