#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market volatility, providing reliable trend signals in both bull and bear markets. Combined with RSI filter to avoid whipsaws, this strategy aims for low trade frequency (target: 25-40/year) on 4h timeframe. Works in trending markets by following KAMA direction, and avoids counter-trend trades during ranging periods via RSI extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Close prices for calculations
    close = prices['close'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.subtract(close, np.roll(close, er_length)))
    vol = np.cumsum(change)
    vol = vol - np.roll(vol, er_length)
    # Avoid division by zero
    er = np.where(vol != 0, dir / vol, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI for filter
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_length] = np.mean(gain[1:rsi_length+1])
    avg_loss[rsi_length] = np.mean(loss[1:rsi_length+1])
    
    for i in range(rsi_length+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i]) / rsi_length
        avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i]) / rsi_length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period average
    volume = prices['volume'].values
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(er_length, rsi_length) + 1, n):
        # Skip if NaN in critical values
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and RSI not overbought
            if price > kama_val and rsi_val < 70 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI not oversold
            elif price < kama_val and rsi_val > 30 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0