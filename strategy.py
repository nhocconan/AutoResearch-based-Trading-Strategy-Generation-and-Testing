#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) as trend filter on daily timeframe,
combined with RSI for momentum confirmation and Choppiness Index to avoid whipsaw in ranging markets.
Long when KAMA slopes up, RSI > 50, and Chop < 61.8 (trending regime); short when KAMA slopes down, RSI < 50, and Chop < 61.8.
Exit when conditions reverse. Uses price close to avoid look-ahead. Designed for low trade frequency (<25/year) to minimize fee drag.
Works in bull/bear: Choppiness filter avoids false signals in range, KAMA adapts to volatility.
"""

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on close (10-period ER, 2/30 fast/slow SC)
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama_vals = np.full(n, np.nan)
        if n < er_period + 1:
            return kama_vals
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.sum(np.abs(np.diff(close)))
        # Manual volatility sum for efficiency
        volatility = np.zeros(n)
        for i in range(er_period, n):
            volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - close[i-er_period:i]))
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constant
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama_vals[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI (14-period)
    def rsi(close, period=14):
        n = len(close)
        rsi_vals = np.full(n, np.nan)
        if n < period + 1:
            return rsi_vals
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate Choppiness Index (14-period)
    def choppiness_index(high, low, close, period=14):
        n = len(close)
        chop = np.full(n, np.nan)
        if n < period + 1:
            return chop
        
        atr = np.zeros(n)
        for i in range(1, n):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # True Range sum over period
        tr_sum = np.zeros(n)
        for i in range(period, n):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros(n)
        min_low = np.zeros(n)
        for i in range(period-1, n):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop = 100 * log10(tr_sum / (max_high - min_low)) / log10(period)
        range_hl = max_high - min_low
        chop = np.where(range_hl > 0, 100 * np.log10(tr_sum / range_hl) / np.log10(period), 50)
        return chop
    
    chop_vals = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope: current vs previous
        kama_slope = kama_vals[i] - kama_vals[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: KAMA up, RSI > 50, Chop < 61.8 (trending)
            if kama_slope > 0 and rsi_vals[i] > 50 and chop_vals[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, Chop < 61.8 (trending)
            elif kama_slope < 0 and rsi_vals[i] < 50 and chop_vals[i] < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR RSI < 50 OR Chop >= 61.8 (range)
            if kama_slope < 0 or rsi_vals[i] < 50 or chop_vals[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI > 50 OR Chop >= 61.8 (range)
            if kama_slope > 0 or rsi_vals[i] > 50 or chop_vals[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals