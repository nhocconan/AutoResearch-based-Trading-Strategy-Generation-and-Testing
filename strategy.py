#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA30 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema30_1d = close_1d.ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema30_1d)
    
    # 1d RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema30_1d_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA30 + RSI > 50 + volume filter
            if (close[i] > ema30_1d_aligned[i] and 
                rsi14_1d_aligned[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA30 + RSI < 50 + volume filter
            elif (close[i] < ema30_1d_aligned[i] and 
                  rsi14_1d_aligned[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA30 OR RSI < 40
            if close[i] < ema30_1d_aligned[i] or rsi14_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA30 OR RSI > 60
            if close[i] > ema30_1d_aligned[i] or rsi14_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA30_RSI_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0