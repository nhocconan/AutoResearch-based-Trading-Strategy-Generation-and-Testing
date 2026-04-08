#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    range_1d = high_1d - low_1d
    close_prev = close_1d  # Using close as the base for pivot
    
    # Camarilla levels: H5 = close + 1.1 * range / 2, L5 = close - 1.1 * range / 2
    camarilla_h5 = close_prev + 1.1 * range_1d / 2
    camarilla_l5 = close_prev - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 12h timeframe
    h5_12h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l5_12h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # 1d trend: 50-period EMA on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h5_12h[i]) or np.isnan(l5_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < L5 or trend fails
            if close[i] < l5_12h[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > H5 or trend fails
            if close[i] > h5_12h[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_12h[i]
            bearish = close[i] < ema_50_12h[i]
            
            # Long: price > H5 + bullish trend + volume
            if (close[i] > h5_12h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < L5 + bearish trend + volume
            elif (close[i] < l5_12h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals