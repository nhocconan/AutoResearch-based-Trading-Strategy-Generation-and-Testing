#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolatility_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    sma50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # 1d data for Camarilla pivot and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels from previous day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d ATR for volatility filter (avoid choppy markets)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(sma50_4h_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, price above 4h SMA50, low volatility
            long_cond = (close[i] > r1_aligned[i] and 
                        close[i] > sma50_4h_aligned[i] and
                        atr14_1d_aligned[i] < np.mean(atr14_1d_aligned[max(0, i-50):i+1]))
            
            # Short: Price breaks below S1, price below 4h SMA50, low volatility
            short_cond = (close[i] < s1_aligned[i] and 
                         close[i] < sma50_4h_aligned[i] and
                         atr14_1d_aligned[i] < np.mean(atr14_1d_aligned[max(0, i-50):i+1]))
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price closes below S1 OR price crosses below 4h SMA50
            if close[i] < s1_aligned[i] or close[i] < sma50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price closes above R1 OR price crosses above 4h SMA50
            if close[i] > r1_aligned[i] or close[i] > sma50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals