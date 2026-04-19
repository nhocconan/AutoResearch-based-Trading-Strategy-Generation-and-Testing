#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Camarilla_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and daily data for Camarilla
    df_w = get_htf_data(prices, '1w')
    df_d = get_htf_data(prices, '1d')
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    volume_d = df_d['volume'].values
    
    # Calculate weekly trend using EMA20 on weekly closes
    ema_20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = ema_20_w[1:] > ema_20_w[:-1]
    weekly_downtrend = ema_20_w[1:] < ema_20_w[:-1]
    weekly_uptrend = np.concatenate([[False], weekly_uptrend])
    weekly_downtrend = np.concatenate([[False], weekly_downtrend])
    
    # Align weekly trend to daily timeframe
    weekly_uptrend_d = align_htf_to_ltf(prices, df_w, weekly_uptrend)
    weekly_downtrend_d = align_htf_to_ltf(prices, df_w, weekly_downtrend)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_periods = len(close_arr)
        R1 = np.full(n_periods, np.nan)
        S1 = np.full(n_periods, np.nan)
        
        for i in range(1, n_periods):
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            R1[i] = close_prev + (high_prev - low_prev) * 1.1 / 12
            S1[i] = close_prev - (high_prev - low_prev) * 1.1 / 12
        return R1, S1
    
    R1_d, S1_d = calculate_camarilla(high_d, low_d, close_d)
    
    # Align Camarilla levels to daily timeframe (already aligned via df_d)
    R1_aligned = align_htf_to_ltf(prices, df_d, R1_d)
    S1_aligned = align_htf_to_ltf(prices, df_d, S1_d)
    
    # Volume spike on daily timeframe
    volume_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_spike_d = volume_d > (volume_ma_d * 2.0)
    volume_spike = align_htf_to_ltf(prices, df_d, volume_spike_d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(weekly_uptrend_d[i]) or np.isnan(weekly_downtrend_d[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly uptrend
            if close[i] > R1_aligned[i] and vol_confirm and weekly_uptrend_d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and weekly downtrend
            elif close[i] < S1_aligned[i] and vol_confirm and weekly_downtrend_d[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit long: price falls below S1 or weekly trend turns down
            if close[i] < S1_aligned[i] or weekly_downtrend_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit short: price rises above R1 or weekly trend turns up
            if close[i] > R1_aligned[i] or weekly_uptrend_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals