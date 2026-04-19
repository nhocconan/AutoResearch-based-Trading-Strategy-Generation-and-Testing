#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_KAMA_Momentum_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate KAMA on 12h close
    close_12h_series = pd.Series(close_12h)
    change = abs(close_12h_series - close_12h_series.shift(10))
    volatility = abs(close_12h_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_12h = [close_12h[0]]
    for i in range(1, len(close_12h)):
        kama_12h.append(kama_12h[-1] + sc.iloc[i] * (close_12h[i] - kama_12h[-1]))
    kama_12h = np.array(kama_12h)
    
    # Align KAMA to 6h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 12h momentum (KAMA slope)
    kama_series = pd.Series(kama_12h_aligned)
    kama_slope = kama_series.diff(5)  # 5-period slope
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_slope.iloc[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when KAMA slope turns positive and price above 1d EMA50 with volume
            if kama_slope.iloc[i] > 0 and close[i] > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when KAMA slope turns negative and price below 1d EMA50 with volume
            elif kama_slope.iloc[i] < 0 and close[i] < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when KAMA slope turns negative or price breaks below EMA50
            if kama_slope.iloc[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when KAMA slope turns positive or price breaks above EMA50
            if kama_slope.iloc[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals