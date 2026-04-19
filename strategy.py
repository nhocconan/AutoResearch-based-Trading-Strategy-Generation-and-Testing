#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_VolumeSpike_ChopRegime_v1"
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
    
    # Get 1d data for TRIX and chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX on 1d close (15-period EMA of EMA of EMA)
    close_1d = df_1d['close'].values
    def ema(arr, period):
        return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    ema1 = ema(close_1d, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Calculate chopiness index on 1d (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        # True range for first element
        tr[0] = high_arr[0] - low_arr[0]
        # ATR calculation
        atr[period-1:] = pd.Series(tr).rolling(window=period, min_periods=period).mean().values[period-1:]
        # Chop calculation
        max_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            if atr[i] > 0:
                chop[i] = 100 * np.log10(sum(tr[i-period+1:i+1]) / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Volume spike indicator (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Align TRIX and chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when TRIX positive, chop indicates trending (< 38.2), volume spike
            if trix_aligned[i] > 0 and chop_aligned[i] < 38.2 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when TRIX negative, chop indicates trending (< 38.2), volume spike
            elif trix_aligned[i] < 0 and chop_aligned[i] < 38.2 and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when TRIX turns negative or chop becomes high (range)
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when TRIX turns positive or chop becomes high (range)
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals