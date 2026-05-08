#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_13_Trend_With_WilliamsVixFix_Extremes"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Direction: Change over er_len periods
    change = np.abs(np.diff(close, n=er_len))
    # Volatility: Sum of absolute changes over er_len periods
    volatility = np.zeros(n)
    for i in range(er_len, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_len+1:i+1])))
    volatility[0:er_len] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    er[er_len:] = change[er_len:] / volatility[er_len:]
    er[0:er_len] = np.nan
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Williams Vix Fix
    # Highest high in 22-day window
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    # WVF = ((Highest High - Low) / Highest High) * 100
    wvf = ((highest_high - low) / highest_high) * 100
    # Extreme levels: oversold > 80, overbought < 20
    wvf_oversold = wvf > 80
    wvf_overbought = wvf < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150  # Sufficient warmup for KAMA and WVF
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(wvf[i]) or 
            np.isnan(wvf_oversold[i]) or np.isnan(wvf_overbought[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + WVF oversold (extreme fear)
            if close[i] > kama[i] and wvf_oversold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + WVF overbought (extreme greed)
            elif close[i] < kama[i] and wvf_overbought[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA (trend change) OR WVF overbought (extreme greed)
            if close[i] < kama[i] or wvf_overbought[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA (trend change) OR WVF oversold (extreme fear)
            if close[i] > kama[i] or wvf_oversold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals