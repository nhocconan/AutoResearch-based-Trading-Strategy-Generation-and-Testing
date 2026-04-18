#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter_V3
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h to determine trend direction, combined with RSI(14) for momentum confirmation and 1d volume spike filter. KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters overextended entries. Volume surge confirms institutional participation. Designed for low trade frequency (<30/year) to minimize fee drag while maintaining edge in both bull and bear markets.
"""

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
    
    # KAMA ( Kaufman Adaptive Moving Average ) parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad the beginning with zeros to match length
    change = np.concatenate([np.zeros(10), change])
    volatility = np.concatenate([np.zeros(10), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) - momentum filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume spike filter (>2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = vol_spike_1d_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, volume spike
            if price > kama_val and rsi_val < 70 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, volume spike
            elif price < kama_val and rsi_val > 30 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Filter_V3"
timeframe = "12h"
leverage = 1.0