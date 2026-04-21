#!/usr/bin/env python3
"""
4h_KAMA_Direction_VolumeConfirm_ATRStop_V2
Hypothesis: 4h KAMA(10,2,30) trend direction + volume spike (>1.8x 20-bar MA) + ATR(14) stoploss (2.0x). 
KAMA adapts to market noise, reducing whipsaw in sideways/choppy markets (common in 2025 BTC/ETH bear/range). 
Volume confirmation ensures breakout legitimacy. ATR stop manages risk. Designed for fewer, higher-quality trades 
(~20-30/year) to overcome fee drag in bear markets. Works in bull (catches trends) and bear (avoids false breaks via 
KAMA's efficiency ratio filtering noise).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA(10,2,30) - Efficiency Ratio based adaptive moving average
    def kama(close, er_fast=2, er_slow=30):
        change = np.abs(np.diff(close, n=10))  # 10-period net change
        vol = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
        # Handle first 10 values
        change = np.concatenate([np.full(10, np.nan), change])
        vol = np.concatenate([np.full(10, np.nan), vol])
        er = np.where(vol != 0, change / vol, 0)  # Efficiency Ratio
        sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2  # Smoothing Constant
        kama = np.full_like(close, np.nan)
        kama[9] = close[9]  # Start after first 10 bars
        for i in range(10, len(close)):
            if np.isnan(kama[i-1]):
                kama[i] = close[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_vals[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # volume spike confirmation
        kama_dir = 1 if kama_vals[i] > kama_vals[i-1] else -1  # KAMA slope direction
        
        if position == 0:
            # Long: KAMA turning up + volume spike
            if kama_dir == 1 and kama_vals[i] > kama_vals[i-1] and vol_ok and price > kama_vals[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + volume spike
            elif kama_dir == -1 and kama_vals[i] < kama_vals[i-1] and vol_ok and price < kama_vals[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or KAMA reverses down with volume
            if price < kama_vals[i] - 2.0 * atr[i] or (kama_dir == -1 and kama_vals[i] < kama_vals[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or KAMA reverses up with volume
            if price > kama_vals[i] + 2.0 * atr[i] or (kama_dir == 1 and kama_vals[i] > kama_vals[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_VolumeConfirm_ATRStop_V2"
timeframe = "4h"
leverage = 1.0