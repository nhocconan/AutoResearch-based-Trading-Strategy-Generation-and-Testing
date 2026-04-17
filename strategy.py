#!/usr/bin/env python3
"""
12h_KAMA_1dRSI_TrendFilter_V1
Trend-following strategy using KAMA direction on 12h with 1d RSI filter for trend strength.
Long when KAMA trending up and 1d RSI > 50; short when KAMA trending down and 1d RSI < 50.
Uses volume confirmation to avoid false signals.
Position size: 0.25. Target: 15-30 trades/year.
Works in bull/bear: KAMA adapts to market noise, RSI filter ensures trend alignment, volume confirms conviction.
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
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio) and SSC (Smoothing Constant) for KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1.0
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 20-period average on 12h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for calculations
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: comparing current KAMA to previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Volume filter: current volume > 20-period average
        volume_filter = volume[i] > volume_ma20[i]
        
        if position == 0:
            # Long: KAMA trending up AND RSI > 50 (bullish momentum)
            if kama_up and rsi_1d_aligned[i] > 50 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down AND RSI < 50 (bearish momentum)
            elif kama_down and rsi_1d_aligned[i] < 50 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_1dRSI_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0