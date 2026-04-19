#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WeeklyTrend_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly trend using EMA50 on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA50 = uptrend
    weekly_downtrend = ema_50_1d[1:] < ema_50_1d[:-1]  # Falling EMA50 = downtrend
    
    # Prepend first value to maintain same length
    weekly_uptrend = np.concatenate([[False], weekly_uptrend])
    weekly_downtrend = np.concatenate([[False], weekly_downtrend])
    
    # Align weekly trend to 4h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1d, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1d, weekly_downtrend)
    
    # Calculate Camarilla pivot levels (R1, S1) on 1d timeframe
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_periods = len(close_arr)
        R1 = np.full(n_periods, np.nan)
        S1 = np.full(n_periods, np.nan)
        
        for i in range(1, n_periods):
            # Use previous period's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Camarilla formulas
            R1[i] = close_prev + (high_prev - low_prev) * 1.1 / 12
            S1[i] = close_prev - (high_prev - low_prev) * 1.1 / 12
        
        return R1, S1
    
    R1_1d, S1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate volume spike indicator (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume AND weekly uptrend
            if close[i] > R1_aligned[i] and vol_confirm and weekly_uptrend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume AND weekly downtrend
            elif close[i] < S1_aligned[i] and vol_confirm and weekly_downtrend_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal) or weekly trend turns down
            if close[i] < S1_aligned[i] or weekly_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal) or weekly trend turns up
            if close[i] > R1_aligned[i] or weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals