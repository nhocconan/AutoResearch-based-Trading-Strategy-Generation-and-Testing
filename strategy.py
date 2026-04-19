#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_Volume_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA calculation on weekly data
    close_1w = df_1w['close'].values
    if len(close_1w) < 10:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1w, n=9))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.concatenate([[0]*9, er])  # pad for first 9 values
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align weekly KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Weekly ATR for volatility filter (14-period)
    tr_w = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr_w = np.maximum(tr_w, np.abs(low[1:] - close[:-1]))
    tr_w = np.concatenate([[np.nan], tr_w])
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_14_w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Volume confirmation: current volume > 1.5x 20-period average (daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_14_w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        atr = atr_14_w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above KAMA with volume and volatility filter
            if price > kama_val and volume_confirmed and atr > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and volatility filter
            elif price < kama_val and volume_confirmed and atr > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or volatility collapse
            if price < kama_val or atr < 0.5 * atr_14_w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or volatility collapse
            if price > kama_val or atr < 0.5 * atr_14_w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals