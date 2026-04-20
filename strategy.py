#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_V1
Hypothesis: Trade KAMA direction with RSI momentum and Choppiness regime filter on daily timeframe.
Long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2); short when KAMA turns down, RSI < 50, and trending.
Uses weekly trend filter to avoid counter-trend trades in bear markets.
Target: 30-100 total trades over 4 years (7-25/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, chop filter avoids ranging markets.
"""

name = "1d_KAMA_RSI_Chop_Filter_V1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA (adaptive moving average)
    def kama(close, window=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, n=window))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[window:] = change[window-1:] / volatility[window-1:]
        er[er == 0] = 1e-10  # avoid division by zero
        
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_val = np.zeros_like(close)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close, 10, 2, 30)
    
    # Calculate RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Calculate Choppiness Index (14-period)
    def choppiness_index(high, low, close, period=14):
        # True range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of true ranges
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral value
        return chop
    
    chop_val = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, trending market (CHOP < 38.2), and above weekly EMA50
            if (kama_val[i] > kama_val[i-1] and rsi_val[i] > 50 and 
                chop_val[i] < 38.2 and close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, trending market (CHOP < 38.2), and below weekly EMA50
            elif (kama_val[i] < kama_val[i-1] and rsi_val[i] < 50 and 
                  chop_val[i] < 38.2 and close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR chop indicates ranging market OR below weekly EMA50
            if (kama_val[i] < kama_val[i-1] or chop_val[i] > 61.8 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR chop indicates ranging market OR above weekly EMA50
            if (kama_val[i] > kama_val[i-1] or chop_val[i] > 61.8 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals