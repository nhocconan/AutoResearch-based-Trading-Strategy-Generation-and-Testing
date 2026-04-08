#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_1w_trend_volume_v1"
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
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels (previous day)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    # Camarilla levels: H4 = close + 1.5*range, L4 = close - 1.5*range
    h4_1d = close_prev + 1.5 * range_1d
    l4_1d = close_prev - 1.5 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    h4_12h = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # 1w trend: 20-period EMA (slower trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 1.5x 30-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < L4 or trend fails (price below weekly EMA)
            if close[i] < l4_12h[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > H4 or trend fails (price above weekly EMA)
            if close[i] > h4_12h[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter from 1w EMA
            bullish = close[i] > ema_20_1w_aligned[i]
            bearish = close[i] < ema_20_1w_aligned[i]
            
            # Long: price > H4 (break above resistance) + bullish trend + volume
            if (close[i] > h4_12h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < L4 (break below support) + bearish trend + volume
            elif (close[i] < l4_12h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals