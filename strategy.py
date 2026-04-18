#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + 1d Trend + Volume Spike
Hypothesis: Williams Fractals identify key pivot points where price reverses. 
Combining with 1d trend direction (via KAMA) filters for breakouts in trend direction.
Volume spike confirms institutional interest. Works in bull/bear by trading breakouts.
Designed for low trade frequency (<50/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_length:
            er[i] = 0
        else:
            change_sum = np.sum(change[i-er_length+1:i+1])
            volatility_sum = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
            if volatility_sum > 0:
                er[i] = change_sum / volatility_sum
            else:
                er[i] = 0
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def williams_fractal(high, low):
    """Calculate Williams Fractals: returns (bearish, bullish) arrays"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
            
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d for trend filter
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_length=10, fast_ema=2, slow_ema=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate Williams Fractals on 4h
    bearish_fractal, bullish_fractal = williams_fractal(high, low)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        kama_val = kama_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: bullish fractal breakout + price above KAMA (uptrend) + volume spike
            if bullish_fractal[i] and close[i] > high[i-1] and close[i] > kama_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal breakout + price below KAMA (downtrend) + volume spike
            elif bearish_fractal[i] and close[i] < low[i-1] and close[i] < kama_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal or price crosses below KAMA
            if bearish_fractal[i] or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal or price crosses above KAMA
            if bullish_fractal[i] or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0