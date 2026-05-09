#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Vortex_Trend_12hEMA50"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Vortex Indicator (VI) on 4h
    period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period)  # Need 50 for EMA50 and 14 for VI
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_12h = ema_50_12h_aligned[i]
        vi_p = vi_plus[i]
        vi_m = vi_minus[i]
        
        if position == 0:
            # Enter long: VI+ > VI- and price above 12h EMA50 (uptrend)
            if vi_p > vi_m and close[i] > ema_12h:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ and price below 12h EMA50 (downtrend)
            elif vi_m > vi_p and close[i] < ema_12h:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- > VI+ (trend change) or price below 12h EMA50
            if vi_m > vi_p or close[i] < ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ > VI- (trend change) or price above 12h EMA50
            if vi_p > vi_m or close[i] > ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals