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
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period high/low on 1d for breakout levels
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period ATR on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / (vol_ma_20_1d + 1e-10)
    
    # Align HTF indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > high_20_aligned[i] and vol_ratio_aligned[i] > 1.5
        breakout_down = close[i] < low_20_aligned[i] and vol_ratio_aligned[i] > 1.5
        
        # Volatility filter: avoid extremely low volatility environments
        vol_filter = atr_14_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Entry conditions
        long_entry = breakout_up and vol_filter
        short_entry = breakout_down and vol_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (close[i] < low_20_aligned[i] or atr_14_aligned[i] < 0.005 * close[i])
        exit_short = position == -1 and (close[i] > high_20_aligned[i] or atr_14_aligned[i] < 0.005 * close[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_breakout_vol_filter"
timeframe = "12h"
leverage = 1.0