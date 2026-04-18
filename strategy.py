#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets. 
Combined with RSI for momentum confirmation and volume filter to avoid false signals, this should work in both bull and bear markets.
Low trade frequency expected due to multiple confirmation requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend filter (10-period efficiency ratio)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        sum_volatility = np.sum(volatility[i-9:i+1])
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0
    # Smooth ER with smoothing constants
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, volume confirmation
            if price > kama_val and rsi_val > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, volume confirmation
            elif price < kama_val and rsi_val < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0