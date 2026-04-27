#!/usr/bin/env python3
"""
4h_RandomWalk_Residual_MeanReversion_v1
Hypothesis: In ranging markets, price deviates from a random walk trend (KAMA) and mean-reverts.
We capture reversions when price deviates >1.5*ATR from KAMA, with volume confirmation and
ADX < 20 to ensure ranging conditions. Works in both bull and bear markets by fading extremes
in consolidation periods.
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
    
    # KAMA (10, 2, 30) as random walk trend
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1], prepend=close[i-9])))
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR (14) for deviation measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX (14) for ranging market detection
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_abs = np.abs(tr)
    atr_adx = pd.Series(tr_abs).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_adx
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_adx
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) > 0, dx, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_avg)
    
    # Deviation from KAMA trend
    deviation = close - kama
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        dev = deviation[i]
        atr_val = atr[i]
        adx_val = adx[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price significantly below KAMA trend (oversold) in ranging market
            if dev < -1.5 * atr_val and adx_val < 20 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price significantly above KAMA trend (overbought) in ranging market
            elif dev > 1.5 * atr_val and adx_val < 20 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price returns to KAMA trend or trend strengthens (ADX > 25)
            if dev > -0.5 * atr_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to KAMA trend or trend strengthens (ADX > 25)
            if dev < 0.5 * atr_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RandomWalk_Residual_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0