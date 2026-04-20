#!/usr/bin/env python3
"""
1d_KAMA_10_RSI14_VolumeSpike_V1
Hypothesis: Daily KAMA(10) trend with RSI(14) mean reversion and volume spike confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets.
Long when KAMA trending up, RSI < 30 (oversold), and volume > 1.5x 20-day average.
Short when KAMA trending down, RSI > 70 (overbought), and volume > 1.5x 20-day average.
Exit when RSI returns to neutral zone (40-60) or trend reverses.
Designed for 1d timeframe to target 7-25 trades/year, avoiding overtrading.
Works in both bull (trend following) and bear (mean reversion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10) on close prices
    # Efficiency ratio: |close - close[9]| / sum(|close[i] - close[i-1]| for i=1..9)
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rsi = np.zeros_like(close)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # No loss = RSI 100
    rsi[:13] = np.nan  # Not enough data
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if NaN in critical values
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        vol_ok = vol_spike[i]
        
        # KAMA trend direction
        kama_up = kama_now > kama_prev
        kama_down = kama_now < kama_prev
        
        if position == 0:
            # Long: KAMA up, RSI oversold, volume spike
            if kama_up and rsi_now < 30 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, volume spike
            elif kama_down and rsi_now > 70 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or KAMA turns down
            if rsi_now > 40 or not kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or KAMA turns up
            if rsi_now < 60 or not kama_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_10_RSI14_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0