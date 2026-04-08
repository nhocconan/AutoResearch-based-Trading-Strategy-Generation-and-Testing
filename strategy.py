#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema200_breakout_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA200 (long-term trend filter)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h EMA20 (entry trigger)
    ema20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < EMA20 or long-term trend fails
            if close[i] < ema20_4h[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > EMA20 or long-term trend fails
            if close[i] > ema20_4h[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume filter: volume > 1.3x 1d average
            vol_filter = volume[i] > (vol_ma_1d_aligned[i] * 1.3)
            
            # Long: price > EMA20 + above long-term EMA200 + volume
            if (close[i] > ema20_4h[i] and 
                close[i] > ema200_1d_aligned[i] and 
                vol_filter):
                position = 1
                signals[i] = 0.25
            # Short: price < EMA20 + below long-term EMA200 + volume
            elif (close[i] < ema20_4h[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals