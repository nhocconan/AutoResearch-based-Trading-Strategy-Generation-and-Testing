#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_with_Volume_and_Chop
Hypothesis: Use KAMA trend direction on daily timeframe as primary filter, combined with volume spike and Choppiness Index regime filter to avoid whipsaws. Designed for low trade frequency (<25/year) on 1d timeframe to minimize fee drag. Works in bull/bear by only taking trades in direction of KAMA trend when market is not too choppy.
"""

name = "1d_KAMA_Trend_Filter_with_Volume_and_Chop"
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
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) on 1d ===
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values  # Replace NaN with 0 when volatility is 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(df_1d['close'], np.nan, dtype=float)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to daily timeframe (no additional delay needed for trend)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Choppiness Index on 1w (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = df_1w['high'].rolling(14, min_periods=14).max().values
    ll = df_1w['low'].rolling(14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.full_like(df_1w['close'], np.nan, dtype=float)
    for i in range(14, len(df_1w)):
        if atr14[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / (atr14[i] * np.log2(14))) / np.log10(hh[i] - ll[i])
        else:
            chop[i] = 50.0  # Neutral when undefined
    
    # Align Choppiness Index to daily timeframe with 1-week delay (wait for weekly close)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === Volume Spike Filter (2x 20-day EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend), low chop (trending market), volume spike
            if (close[i] > kama_aligned[i] and 
                chop_aligned[i] < 50.0 and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA (downtrend), low chop (trending market), volume spike
            elif (close[i] < kama_aligned[i] and 
                  chop_aligned[i] < 50.0 and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses KAMA in opposite direction OR chop becomes too high (choppy market)
            if position == 1:
                if close[i] < kama_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > kama_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals