#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Vortex_Trend_Filter_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI) on 1d
    # True Range
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[:-1])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +VM and -VM
    vm_plus = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[:-1])
    vm_minus = np.abs(df_1d['low'].values[1:] - df_1d['high'].values[:-1])
    vm_plus = np.concatenate([[0], vm_plus])
    vm_minus = np.concatenate([[0], vm_minus])
    
    # Sum over 14 periods (standard VI period)
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    vi_plus_4h = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_4h = align_htf_to_ltf(prices, df_1d, vi_minus)
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_1d_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus_4h[i]) or np.isnan(vi_minus_4h[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(vol_avg_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vi_plus_val = vi_plus_4h[i]
        vi_minus_val = vi_minus_4h[i]
        trend = ema50_12h_4h[i]
        vol_avg = vol_avg_1d_4h[i]
        vol_ok = volume[i] > vol_avg * 2.0  # Higher threshold for fewer trades
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) + volume + price above trend
            if vi_plus_val > vi_minus_val and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) + volume + price below trend
            elif vi_minus_val > vi_plus_val and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or VI crossover
            if close[i] < trend or vi_minus_val > vi_plus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or VI crossover
            if close[i] > trend or vi_plus_val > vi_minus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals