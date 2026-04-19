#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Adaptive_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Adaptive Moving Average) - 4h
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = change[10]  # avoid NaN for first 10
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder for correct calc
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = np.concatenate([[0], np.abs(np.diff(close))])  # volatility per bar
    # Sum volatility over 10 periods
    vol_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        vol_sum[i] = np.sum(volatility[i-9:i+1])
    vol_sum[0:10] = vol_sum[10]
    er = np.where(vol_sum != 0, change / vol_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - 4h
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily ATR for regime filter (chop detection)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # Normalize ATR by price to get churn
    atr_norm = atr_1d / close_1d
    # Choppiness-like: high ATR_norm = choppy, low = trending
    atr_norm_ma = pd.Series(atr_norm).rolling(window=10, min_periods=10).mean().values
    atr_norm_aligned = align_htf_to_ltf(prices, df_1d, atr_norm_ma)
    
    # Align KAMA and RSI to 4h (they are already 4h, but for consistency)
    kama_aligned = kama  # already 4h
    rsi_aligned = rsi    # already 4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_norm_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = atr_norm_aligned[i]
        
        # Regime filter: only trade when trending (low churn)
        trending = chop_val < 0.02  # threshold for trending regime
        
        if position == 0:
            # Long: price above KAMA and RSI > 50 in trending market
            if price > kama_val and rsi_val > 50 and trending:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI < 50 in trending market
            elif price < kama_val and rsi_val < 50 and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals