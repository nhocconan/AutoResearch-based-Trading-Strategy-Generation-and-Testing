#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_rsi_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    kama_up = kama_slope > 0
    kama_down = kama_slope < 0
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rs), 50, rsi)
    
    # Chop index (14) for regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest - lowest)) / np.log10(14)
    chop = np.where((highest - lowest) == 0, 50, chop)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Volume confirmation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR chop too high (trending breaks)
            if kama_down[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR chop too high
            if kama_up[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume and chop not extreme (avoid choppy markets)
            if not volume_ok[i] or chop[i] > 61.8:
                signals[i] = 0.0
                continue
            
            # Long: KAMA up + RSI not overbought
            if kama_up[i] and rsi[i] < 70:
                position = 1
                signals[i] = 0.25
            # Short: KAMA down + RSI not oversold
            elif kama_down[i] and rsi[i] > 30:
                position = -1
                signals[i] = -0.25
    
    return signals