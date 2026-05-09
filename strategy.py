#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
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
    
    # Camarilla levels from previous day
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    range_val = prev_high - prev_low
    r1 = prev_close + (range_val * 1.1 / 12)
    s1 = prev_close - (range_val * 1.1 / 12)
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_avg20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume spike
            if (price > r1[i] and 
                price > ema50_12h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 + 12h downtrend + volume spike
            elif (price < s1[i] and 
                  price < ema50_12h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns below S1 or 12h trend fails
            if (price < s1[i] or 
                price < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or 12h trend fails
            if (price > r1[i] or 
                price > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals