#!/usr/bin/env python3
name = "4h_ParabolicSAR_EMA13_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Parabolic SAR (AF=0.02, max=0.2) ===
    # Initialize
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    ep = high[0] if trend == 1 else low[0]
    
    for i in range(1, n):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        
        if trend == 1:
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # === EMA13 ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Volume trend (20-period EMA of volume) ===
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema  # Current volume relative to average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: Price above PSAR AND above EMA13 AND volume above average
            if close[i] > psar[i] and close[i] > ema13[i] and vol_ratio[i] > 1.2:
                signals[i] = 0.25
                position = 1
            # Short: Price below PSAR AND below EMA13 AND volume above average
            elif close[i] < psar[i] and close[i] < ema13[i] and vol_ratio[i] > 1.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below PSAR OR below EMA13
            if close[i] < psar[i] or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above PSAR OR above EMA13
            if close[i] > psar[i] or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals