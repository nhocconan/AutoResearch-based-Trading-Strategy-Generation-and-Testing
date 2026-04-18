#!/usr/bin/env python3
"""
4h_VolumeBreakout_KAMA_Trend_v1
Hypothesis: Volume breakouts above KAMA(14) trend with ATR filter capture momentum in both bull and bear markets.
KAMA adapts to market noise, reducing false signals. Volume breakout confirms conviction. Designed for ~30 trades/year.
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
    
    # KAMA(14) trend filter
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=14, min_periods=14).sum()
    er = change.rolling(window=14, min_periods=14).sum() / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = abs(high[1:] - close[:-1])
    tr3 = abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(atr[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above KAMA with volume spike and volatility filter
            if price > kama_val and vol_spike and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike and volatility filter
            elif price < kama_val and vol_spike and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to KAMA or volatility drops
            if price < kama_val or atr_val < atr[i-1] * 0.5:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to KAMA or volatility drops
            if price > kama_val or atr_val < atr[i-1] * 0.5:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_VolumeBreakout_KAMA_Trend_v1"
timeframe = "4h"
leverage = 1.0