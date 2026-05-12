#!/usr/bin/env python3
name = "6h_ADX_Alligator_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) on 1d
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Alligator on 1d: SMMA (Jaw=13, Teeth=8, Lips=5)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        result[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = sma[i]
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Align to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike on 6h: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + Lips > Teeth > Jaw (bullish alignment) + volume spike
            if (adx_aligned[i] > 25 and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) + Lips < Teeth < Jaw (bearish alignment) + volume spike
            elif (adx_aligned[i] > 25 and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX drops below 20 (weakening trend) or alignment breaks
            if (adx_aligned[i] < 20 or 
                lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX drops below 20 (weakening trend) or alignment breaks
            if (adx_aligned[i] < 20 or 
                lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals