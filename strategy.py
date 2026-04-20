#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily KAMA for trend direction ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === Daily RSI for momentum ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily Chopiness Index for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Align to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, chop < 61.8 (trending), volume confirmation
            if close_val > kama_val and rsi_val > 50 and chop_val < 61.8 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, chop < 61.8 (trending), volume confirmation
            elif close_val < kama_val and rsi_val < 50 and chop_val < 61.8 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below KAMA or RSI < 40 or chop > 61.8 (ranging)
            if close_val < kama_val or rsi_val < 40 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above KAMA or RSI > 60 or chop > 61.8 (ranging)
            if close_val > kama_val or rsi_val > 60 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals