#!/usr/bin/env python3
# 4h_1d_kama_rsi_v1
# Strategy: 4h KAMA direction + RSI(14) filter + 1d volume spike confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in ranging markets. 
# RSI filters extreme momentum, while 1d volume spike confirms institutional interest.
# Works in bull (KAMA up + RSI > 50) and bear (KAMA down + RSI < 50) with volume confirmation.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    vol_1d_series = pd.Series(vol_1d)
    vol_avg_20_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # KAMA calculation: adaptive moving average
    # Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_change = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.where(sum_abs_change > 0, change / sum_abs_change, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Entry logic: KAMA direction + RSI filter + 1d volume spike
        if (price_above_kama and rsi[i] > 50 and vol_spike_1d_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (price_below_kama and rsi[i] < 50 and vol_spike_1d_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA cross or RSI extreme
        elif position == 1 and (price_below_kama or rsi[i] < 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price_above_kama or rsi[i] > 70):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals