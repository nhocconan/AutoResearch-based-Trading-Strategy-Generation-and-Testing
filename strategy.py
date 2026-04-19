#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, 1))
    er = change / (volatility + 1e-10)
    er = np.where(np.isnan(er), 0, er)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)
    
    # Calculate 1d Choppiness Index
    atr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    atr2 = np.abs(high_1d[1:] - close_1d[:-1])
    atr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(atr1, np.maximum(atr2, atr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h volume filter: current volume > 1.5x 24-period average (12d)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Choppiness filter: avoid extreme chop (>61.8) or strong trend (<38.2)
        chop_val = chop_aligned[i]
        chop_ok = 38.2 <= chop_val <= 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, chop OK, volume OK
            if price > kama_aligned[i] and rsi_aligned[i] > 50 and chop_ok and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, chop OK, volume OK
            elif price < kama_aligned[i] and rsi_aligned[i] < 50 and chop_ok and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price < KAMA or reverse signal
            if price < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > kama_aligned[i] and rsi_aligned[i] < 30 and chop_ok and volume_ok:
                # Reverse to short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price > KAMA or reverse signal
            if price > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < kama_aligned[i] and rsi_aligned[i] > 70 and chop_ok and volume_ok:
                # Reverse to long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals