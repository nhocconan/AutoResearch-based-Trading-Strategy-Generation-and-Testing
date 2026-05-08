#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_CCI_Trend_Filter_With_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly CCI(20) for trend filter
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    sma_tp_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    mad_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_20_1w = (tp_1w - sma_tp_1w) / (0.015 * mad_1w)
    cci_20_1w = np.nan_to_num(cci_20_1w, nan=0.0)
    cci_20_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_20_1w)
    
    # Daily CCI(20) for entry signal
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_20 = (tp - sma_tp) / (0.015 * mad)
    cci_20 = np.nan_to_num(cci_20, nan=0.0)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(cci_20_1w_aligned[i]) or np.isnan(cci_20[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI > 100, weekly CCI > 0, volume spike
            long_cond = (cci_20[i] > 100 and 
                        cci_20_1w_aligned[i] > 0 and
                        volume_spike[i])
            
            # Short: CCI < -100, weekly CCI < 0, volume spike
            short_cond = (cci_20[i] < -100 and 
                         cci_20_1w_aligned[i] < 0 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI crosses below 0
            if cci_20[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI crosses above 0
            if cci_20[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals