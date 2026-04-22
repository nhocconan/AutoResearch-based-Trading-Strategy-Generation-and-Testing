# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h KAMA with RSI confirmation and volume spike filter.
# Uses adaptive KAMA to filter noise and RSI(14) for momentum confirmation.
# Volume spike ensures participation. Designed for 6h to capture multi-day swings
# with low frequency in both bull and bear markets. Target: 20-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, FAST = 2, SLOW = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if vol[i] != 0:
            er[i] = change[i] / vol[i]
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (24-period on 6h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma24
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 + volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI < 50 + volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses KAMA or RSI reverses
            if position == 1:
                if (close[i] < kama[i] or rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > kama[i] or rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_KAMA_RSI_Volume_Spike"
timeframe = "6h"
leverage = 1.0