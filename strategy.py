#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === KAMA for trend detection ===
    close = prices['close'].values
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / np.where(volatility > 0, volatility, 1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI for momentum filter ===
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # === Chopiness Index for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    atr = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr = np.maximum(atr, np.maximum(tr1, tr2))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_period = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    high_max = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_min = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.where(high_max - low_min > 0, high_max - low_min, 1)) / np.log10(14)
    
    # === KAMA direction (from daily) ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    change_1d = abs(close_1d_series.diff(1))
    volatility_1d = change_1d.rolling(window=10, min_periods=10).sum()
    er_1d = change_1d.rolling(window=10, min_periods=10).sum() / np.where(volatility_1d > 0, volatility_1d, 1)
    sc_1d = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_dir = kama_1d > np.roll(kama_1d, 1)
    kama_1d_dir[0] = True
    kama_1d_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_dir.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_1d_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi_values[i]
        chop_val = chop[i]
        kama_1d_dir_val = kama_1d_dir_aligned[i]
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending), daily KAMA up
            if kama_val > close[i] and rsi_val > 50 and chop_val < 61.8 and kama_1d_dir_val > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, chop < 61.8 (trending), daily KAMA down
            elif kama_val < close[i] and rsi_val < 50 and chop_val < 61.8 and kama_1d_dir_val < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR chop > 61.8 (choppy)
            if kama_val < close[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR chop > 61.8 (choppy)
            if kama_val > close[i] or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals