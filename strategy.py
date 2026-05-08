#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_CCI_Trend_Filter_With_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily CCI(20)
    tp = (high_1d + low_1d + close_1d) / 3
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    cci = cci.fillna(0).values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci[i-1]) if i-1 >= 0 else True or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI > 100, price above EMA50, volume spike
            long_cond = (cci[i-1] > 100 and 
                        close[i] > ema_50_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: CCI < -100, price below EMA50, volume spike
            short_cond = (cci[i-1] < -100 and 
                         close[i] < ema_50_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI < 0 or price crosses below EMA50
            if cci[i-1] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI > 0 or price crosses above EMA50
            if cci[i-1] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals