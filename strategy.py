#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and R1/S1 levels (using previous day's data)
    def calculate_pivots(high_arr, low_arr, close_arr):
        n_days = len(close_arr)
        pivot = np.full(n_days, np.nan)
        R1 = np.full(n_days, np.nan)
        S1 = np.full(n_days, np.nan)
        
        for i in range(1, n_days):
            # Use previous day's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Standard pivot point calculation
            pivot[i] = (high_prev + low_prev + close_prev) / 3.0
            # R1 and S1 levels
            R1[i] = 2 * pivot[i] - low_prev
            S1[i] = 2 * pivot[i] - high_prev
        
        return pivot, R1, S1
    
    pivot_1d, R1_1d, S1_1d = calculate_pivots(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike indicator (volume > 1.8 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        # Volatility filter: only trade when ATR is above average
        vol_filter = atr[i] > np.nanmedian(atr[max(0, i-50):i+1]) if i >= 50 else True
        
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume and volatility
            if close[i] > R1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume and volatility
            elif close[i] < S1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below pivot (reversal)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above pivot (reversal)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals