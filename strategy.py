#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR(14) for volatility and position sizing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Daily ATR-based volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_1d_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_1d_aligned < atr_median
    
    # Volume filter: volume > 1.8x 20-period average (to avoid false breakouts)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA200 + volume filter + low volatility
            if (close[i] > ema200_1w_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA200 + volume filter + low volatility
            elif (close[i] < ema200_1w_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA200 (trend change)
            if close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly EMA200 (trend change)
            if close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA200_Volume_LowVol_Filter_v1"
timeframe = "12h"
leverage = 1.0