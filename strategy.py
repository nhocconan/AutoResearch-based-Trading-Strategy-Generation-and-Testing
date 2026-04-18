#!/usr/bin/env python3
"""
4h_KAMA_Trend_Reversal_with_RSI_Filter
Hypothesis: KAMA adapts to market noise; when price crosses KAMA with RSI confirmation (50/50 cross) and volume spike, it signals a trend reversal. Works in bull/bear by catching mean-reversion moves in ranging markets and momentum shifts in trends. Low trade frequency via volume spike filter.
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.sum(np.abs(np.diff(close)))
    # Vectorized volatility calculation
    volatility = np.zeros_like(change)
    for i in range(er_len, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_len:i])))
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.zeros(er_len), er])
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Volume spike > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14*2)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, volume spike
            if price > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, volume spike
            elif price < kama_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_Reversal_with_RSI_Filter"
timeframe = "4h"
leverage = 1.0