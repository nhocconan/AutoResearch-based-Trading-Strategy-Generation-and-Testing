#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI14_Pullback_Volume_Spike
Hypothesis: KAMA (Kaufman Adaptive Moving Average) trend direction on 4h combined with RSI14 pullback and volume spike provides high-probability entries in both bull and bear markets. KAMA adapts to market noise, reducing whipsaw in ranging conditions while capturing trends. RSI14 pullback to 40-60 area ensures entries occur during temporary pullbacks within the trend, improving risk-reward. Volume spike confirms institutional participation. Target: 20-40 trades/year per symbol.
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.maximum(volatility[length-1:], 1e-10)
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, length=14)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = vol_ma_20[10]  # fill beginning
    vol_ma_20[-10:] = vol_ma_20[-11]  # fill end
    volume_spike = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction
        kama_up = close[i] > kama_vals[i]
        kama_down = close[i] < kama_vals[i]
        
        # RSI pullback condition: RSI between 40 and 60 (not overbought/oversold)
        rsi_pullback = (rsi_vals[i] >= 40) & (rsi_vals[i] <= 60)
        
        # Entry conditions
        long_entry = kama_up and rsi_pullback and volume_spike[i]
        short_entry = kama_down and rsi_pullback and volume_spike[i]
        
        # Exit on opposite KAMA crossover (reverse position)
        long_exit = kama_down and volume_spike[i]
        short_exit = kama_up and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI14_Pullback_Volume_Spike"
timeframe = "4h"
leverage = 1.0