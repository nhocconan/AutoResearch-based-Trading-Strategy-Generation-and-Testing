#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA(50)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA(200)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 1d timeframe (already aligned as 1d is the base timeframe)
    ema_50_1d_aligned = ema_50_1d
    ema_200_1d_aligned = ema_200_1d
    atr_14_1d_aligned = atr_14_1d
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price above EMA200 and volatility is below 50th percentile
            if price > ema_200_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 and volatility is below 50th percentile
            elif price < ema_50_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA50 or volatility above 70th percentile
            if price < ema_50_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above EMA200 or volatility above 70th percentile
            if price > ema_200_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA50_EMA200_VolatilityFilter"
timeframe = "1d"
leverage = 1.0