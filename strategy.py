#!/usr/bin/env python3
"""
6H Trend Reversal with RSI Momentum and Volume Confirmation
Long when RSI crosses above 30 from below with volume spike AND price above 100 EMA
Short when RSI crosses below 70 from above with volume spike AND price below 100 EMA
Exit when RSI crosses 70 (long) or 30 (short)
Designed to capture reversals in both bull and bear markets with strict entry filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_reversal_volume_ema100_filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === EMA 100 filter ===
    ema_100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or 
            np.isnan(ema_100[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 70
            if rsi[i] >= 70 and rsi[i-1] < 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 30
            if rsi[i] <= 30 and rsi[i-1] > 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            # Long: RSI crosses above 30 from below AND price above EMA100
            if rsi[i] > 30 and rsi[i-1] <= 30 and close[i] > ema_100[i]:
                position = 1
                signals[i] = 0.25
            # Short: RSI crosses below 70 from above AND price below EMA100
            elif rsi[i] < 70 and rsi[i-1] >= 70 and close[i] < ema_100[i]:
                position = -1
                signals[i] = -0.25
    
    return signals