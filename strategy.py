#!/usr/bin/env python3
"""
4h_4hKAMA_1dRSI_TrendFilter_v1
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h captures trend direction with low lag, while 1d RSI filters for overextended conditions. Only take long when KAMA trending up AND RSI < 70 (not overbought), short when KAMA trending down AND RSI > 30 (not oversold). Uses volume confirmation to avoid false signals. Designed for 4-6 trades/year per symbol with tight entries to minimize fee drag. Works in bull/bear by following adaptive trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast_len=2, slow_len=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.sum(np.abs(np.diff(close)))
    # Handle 1D case
    if len(change.shape) == 1:
        volatility = np.array([np.sum(np.abs(np.diff(close[j-er_len:j])) if j >= er_len else 0 for j in range(len(close)))])
    else:
        volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, length=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) > length:
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for KAMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 4h close
    close_4h = df_4h['close'].values
    kama_4h = calculate_kama(close_4h, er_len=10, fast_len=2, slow_len=30)
    kama_4h_slope = np.diff(kama_4h, prepend=kama_4h[0])  # slope = change
    kama_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_slope)
    
    # Load 1d data once for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 1d close
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for indicators to stabilize
        # Skip if indicators not ready
        if np.isnan(kama_4h_slope_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.1 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.1 * vol_ma
        else:
            volume_ok = False
        
        kama_slope = kama_4h_slope_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        if position == 0:
            # Long: KAMA trending up (positive slope) AND RSI not overbought (<70) AND volume
            if kama_slope > 0 and rsi < 70 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down (negative slope) AND RSI not oversold (>30) AND volume
            elif kama_slope < 0 and rsi > 30 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA trend changes or RSI overbought
            if kama_slope <= 0 or rsi >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA trend changes or RSI oversold
            if kama_slope >= 0 or rsi <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4hKAMA_1dRSI_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0