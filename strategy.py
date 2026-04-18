#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Pullback
Hypothesis: KAMA(14) captures adaptive trend direction; RSI(14) pullbacks to KAMA provide
high-probability entries in both bull and bear markets. Uses volume confirmation to filter
false signals. Designed for low trade frequency (15-30/year) to minimize fee drag while
capturing trend continuations. KAMA adapts to market noise, reducing whipsaw in chop.
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
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_conf[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # Long: price pulls back to KAMA in uptrend (RSI < 40) with volume
            if price > kama_val and rsi_val < 40 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to KAMA in downtrend (RSI > 60) with volume
            elif price < kama_val and rsi_val > 60 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI > 70 (overbought)
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI < 30 (oversold)
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_RSI_Pullback"
timeframe = "4h"
leverage = 1.0