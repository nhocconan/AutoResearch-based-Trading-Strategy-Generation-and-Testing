#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_Pullback
Hypothesis: Buy when price pulls back to KAMA(10) in a rising trend (KAMA rising) with RSI < 40; short when price pulls back to KAMA in a falling trend (KAMA falling) with RSI > 60. Uses KAMA for adaptive trend following and RSI for mean-reversion entries. Designed for low trade frequency (<50/year) to minimize fee drag while capturing high-probability mean-reversion moves within the trend. Works in both bull and bear markets by trading with the trend on pullbacks.
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
    fast_sc = 2 / (2 + 1)   # EMA(2) for fast
    slow_sc = 2 / (30 + 1)  # EMA(30) for slow
    
    # Calculate Efficiency Ratio and smoothing constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, n):  # ER needs 10 periods
        if i >= 10:
            net_change = np.abs(close[i] - close[i-10])
            total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # need RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_conf[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_conf = volume_conf[i]
        
        # Trend detection: KAMA slope over 3 periods
        if i >= 3:
            kama_slope = kama[i] - kama[i-3]
        else:
            kama_slope = 0
        
        if position == 0:
            # Long: pullback to KAMA in uptrend with oversold RSI
            if kama_slope > 0 and price <= kama_val * 1.005 and rsi_val < 40 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: pullback to KAMA in downtrend with overbought RSI
            elif kama_slope < 0 and price >= kama_val * 0.995 and rsi_val > 60 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend change or overbought
            if kama_slope <= 0 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend change or oversold
            if kama_slope >= 0 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_RSI_Trend_Pullback"
timeframe = "4h"
leverage = 1.0