#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for 12h strategy
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's values for today's calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Daily ATR for volatility filter
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                               abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Daily volume average
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    
    # Align 1d data to 12h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d.values)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Daily range
        daily_range = high_1d_aligned[i] - low_1d_aligned[i]
        
        # Calculate Camarilla levels from previous day's data
        camarilla_pp = (high_1d_aligned[i] + low_1d_aligned[i] + close_1d_aligned[i]) / 3
        camarilla_r4 = camarilla_pp + (daily_range * 1.1 / 2)
        camarilla_s4 = camarilla_pp - (daily_range * 1.1 / 2)
        
        # Volatility filter: only trade when daily ATR is above average
        vol_filter = atr_1d_aligned[i] > vol_avg_1d_aligned[i] * 0.5
        
        # Volume filter: current daily volume > 1.5x 20-day average
        vol_condition = vol_1d_aligned[i] > vol_avg_1d_aligned[i] * 1.5
        
        # Entry conditions
        long_entry = (close[i] > camarilla_r4) and vol_filter and vol_condition
        short_entry = (close[i] < camarilla_s4) and vol_filter and vol_condition
        
        # Exit conditions: price crosses daily pivot point
        long_exit = close[i] < camarilla_pp
        short_exit = close[i] > camarilla_pp
        
        if position == 0:
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_And_Volatility_Filter"
timeframe = "12h"
leverage = 1.0