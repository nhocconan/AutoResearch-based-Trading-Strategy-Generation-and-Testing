#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_OverboughtOversold_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and choppy markets. 
Combined with RSI extremes (overbought/oversold) and volume confirmation, this captures mean-reversion within the trend.
Exit when RSI returns to neutral zone (40-60). Designed for fewer, higher-quality trades.
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
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_changes = 0
            for j in range(1, 11):
                sum_abs_changes += np.abs(close[i-j+1] - close[i-j])
            if sum_abs_changes > 0:
                er[i] = price_change / sum_abs_changes
            else:
                er[i] = 0
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    sc[0:10] = 0  # initialize
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x average volume
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = volume[:20].mean() if len(volume) >= 20 else volume.mean()
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA, RSI oversold (<30), volume confirmation
            if price > kama_val and rsi_val < 30 and vol_filt:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI overbought (>70), volume confirmation
            elif price < kama_val and rsi_val > 70 and vol_filt:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: hold until RSI returns to neutral (40-60) or price crosses below KAMA
            if 40 <= rsi_val <= 60 or price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: hold until RSI returns to neutral (40-60) or price crosses above KAMA
            if 40 <= rsi_val <= 60 or price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_OverboughtOversold_v1"
timeframe = "4h"
leverage = 1.0