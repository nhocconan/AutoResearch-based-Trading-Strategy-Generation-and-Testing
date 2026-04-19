#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    fast = 2
    slow = 30
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).cumsum()
    volatility = np.concatenate([[0], volatility[1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align daily KAMA to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 4h data (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index on 4h data (14-period)
    atr_1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    atr_1 = np.maximum(atr_1, np.abs(low[1:] - close[:-1]))
    atr_1 = np.concatenate([[0], atr_1])
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        # Chop filter: only trade when not in strong chop (Chop > 61.8 indicates ranging)
        not_choppy = chop_val < 61.8
        
        if position == 0:
            # Long: price above KAMA and RSI > 50 with volume and not choppy
            if price > kama_val and rsi_val > 50 and volume_confirmed and not_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI < 50 with volume and not choppy
            elif price < kama_val and rsi_val < 50 and volume_confirmed and not_choppy:
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