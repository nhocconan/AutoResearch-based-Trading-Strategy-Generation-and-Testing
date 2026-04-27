#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d EMA10 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema10_1d = close_1d.ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # 1d ATR10 for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for EMA10 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema10_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA10 + high volume + low volatility
            if (close[i] > ema10_1d_aligned[i] and 
                volume_filter[i] and 
                atr10_1d_aligned[i] < np.nanmedian(atr10_1d_aligned[:i+1]) and
                atr10_1d_aligned[i] > 0):  # ensure valid ATR
                signals[i] = 0.25
                position = 1
            # Short: price < EMA10 + high volume + low volatility
            elif (close[i] < ema10_1d_aligned[i] and 
                  volume_filter[i] and 
                  atr10_1d_aligned[i] < np.nanmedian(atr10_1d_aligned[:i+1]) and
                  atr10_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA10 (trend change)
            if close[i] < ema10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > EMA10 (trend change)
            if close[i] > ema10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA10_Vol_LowVol_Filter"
timeframe = "4h"
leverage = 1.0