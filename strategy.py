#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_Follow_With_1d_RSI_Filter"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    # KAMA parameters: fast=2, slow=30 (common values)
    fast_sc = 2
    slow_sc = 30
    er_period = 10
    
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d))
    change = np.insert(change, 0, np.nan)  # align length
    dir_1d = np.abs(np.diff(close_1d, lag=er_period))
    dir_1d = np.insert(dir_1d, 0, np.nan)  # align length
    
    volatility = np.nansum(change.reshape(-1, er_period), axis=1)
    volatility = np.insert(volatility, 0, np.nan)  # align length
    
    er = np.divide(dir_1d, volatility, out=np.full_like(dir_1d, np.nan), where=volatility!=0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama_1d[i-1]):
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
        else:
            kama_1d[i] = kama_1d[i-1]
    
    # Calculate 1d RSI for filter
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, np.nan)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    rsi_len = 14
    avg_up = np.full_like(close_1d, np.nan)
    avg_down = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing for RSI
    if len(up) >= rsi_len:
        avg_up[rsi_len-1] = np.nanmean(up[1:rsi_len+1])
        avg_down[rsi_len-1] = np.nanmean(down[1:rsi_len+1])
        
        for i in range(rsi_len, len(up)):
            avg_up[i] = (avg_up[i-1] * (rsi_len-1) + up[i]) / rsi_len
            avg_down[i] = (avg_down[i-1] * (rsi_len-1) + down[i]) / rsi_len
    
    rs = np.divide(avg_up, avg_down, out=np.full_like(avg_up, np.nan), where=avg_down!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h price relative to KAMA
    price_above_kama = close > kama_1d_aligned
    price_below_kama = close < kama_1d_aligned
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = price_above_kama[i] and rsi_1d_aligned[i] > 50 and volume_ok[i]
        short_entry = price_below_kama[i] and rsi_1d_aligned[i] < 50 and volume_ok[i]
        
        # Exit conditions
        long_exit = not price_above_kama[i] or rsi_1d_aligned[i] < 50
        short_exit = not price_below_kama[i] or rsi_1d_aligned[i] > 50
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals