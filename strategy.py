#!/usr/bin/env python3
name = "1d_1w_Turtle_Trend_Reversal"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d Donchian channels (20-day breakout)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w ATR for volatility filter (ATR > 20-period average indicates high volatility)
    def calculate_atr(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First TR
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # 1d volume spike: > 2.0x 20-period average
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume > 2.0 * vol_ma_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_1w_aligned[i]) or
            np.isnan(vol_ma_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-day high with high volatility (ATR > MA) and volume spike
            if (high[i] > high_20[i] and atr_1w_aligned[i] > atr_ma_1w_aligned[i] and vol_spike_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low with high volatility (ATR > MA) and volume spike
            elif (low[i] < low_20[i] and atr_1w_aligned[i] > atr_ma_1w_aligned[i] and vol_spike_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below 10-day low or volatility drops
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if (np.isnan(low_10[i]) or close[i] < low_10[i] or atr_1w_aligned[i] < atr_ma_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above 10-day high or volatility drops
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if (np.isnan(high_10[i]) or close[i] > high_10[i] or atr_1w_aligned[i] < atr_ma_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals