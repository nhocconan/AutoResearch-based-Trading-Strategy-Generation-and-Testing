#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyKAMA_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and KAMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = abs(df_1w['close'].diff(er_period)).values
    volatility = abs(df_1w['close'].diff()).rolling(window=er_period).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_1w['close'])
    kama[0] = df_1w['close'].iloc[0]
    for i in range(1, len(df_1w)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'].iloc[i] - kama[i-1])
    
    # Get daily volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after enough data for calculations
    start_idx = max(20, er_period)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = kama_aligned[i]
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 0:
            # Enter long when price above KAMA with volume confirmation
            if close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when price below KAMA with volume confirmation
            elif close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below KAMA
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above KAMA
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals