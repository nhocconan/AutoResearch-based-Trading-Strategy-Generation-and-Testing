#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_RSI14_ChoppinessFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1d data for Choppiness index
    df_1d_chop = df_1d.copy()
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    close_1d = df_1d['close'].values
    # ER (Efficiency Ratio)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else np.array([0.0])
    volatility = np.concatenate([[0.0], np.abs(np.diff(close_1d))])
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            if volatility[i] != 0:
                er[i] = np.abs(close_1d[i] - close_1d[i-10]) / volatility[i-9:i+1].sum() if i >= 10 else 0
            else:
                er[i] = 0
    # Smooth ER
    er = pd.Series(er).rolling(window=10, min_periods=1).mean().values
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on 1d
    atr_1d = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(df_1d['high'].values, df_1d['low'].values))
    tr2 = np.abs(np.subtract(df_1d['high'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr3 = np.abs(np.subtract(df_1d['low'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    high_roll = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = np.where((high_roll - low_roll) != 0, 
                    100 * np.log10(np.sum(atr_1d) / (high_roll - low_roll)) / np.log10(14), 
                    50)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 4h
    kama_1d = kama
    rsi_1d = rsi
    chop_1d = chop
    
    kama_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = kama_4h[i]
        rsi_val = rsi_4h[i]
        chop_val = chop_4h[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, choppy market (chop > 61.8)
            if close[i] > trend and rsi_val > 50 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, choppy market (chop > 61.8)
            elif close[i] < trend and rsi_val < 50 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40
            if close[i] < trend or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60
            if close[i] > trend or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals