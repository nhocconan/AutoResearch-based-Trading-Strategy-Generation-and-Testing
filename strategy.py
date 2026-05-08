#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Trend_1dVolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0.0], tr])  # align length
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize ATR by price to get relative volatility
    atr_pct = atr / close_1d
    atr_pct_ma = pd.Series(atr_pct).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_pct / atr_pct_ma  # current volatility vs average
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate EMA on 4h data for trend
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fast EMA above slow EMA + volatility above average
            if (ema_fast[i] > ema_slow[i] and vol_ratio_aligned[i] > 1.1):
                signals[i] = 0.25
                position = 1
            # Short: Fast EMA below slow EMA + volatility above average
            elif (ema_fast[i] < ema_slow[i] and vol_ratio_aligned[i] > 1.1):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fast EMA crosses below slow EMA
            if ema_fast[i] < ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fast EMA crosses above slow EMA
            if ema_fast[i] > ema_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals