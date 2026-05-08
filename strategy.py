#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and Alligator indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (14-period)
    period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[period] = np.mean(tr[1:period+1])
    dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
    
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    mask = atr != 0
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    
    di_sum = di_plus + di_minus
    mask = di_sum != 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / di_sum[mask]
    
    # ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Williams Alligator on daily data
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    med_price_1d = (high_1d + low_1d) / 2
    
    def sma(arr, window):
        return np.convolve(arr, np.ones(window)/window, mode='same')
    
    jaw = sma(med_price_1d, jaw_period)
    teeth = sma(med_price_1d, teeth_period)
    lips = sma(med_price_1d, lips_period)
    
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: all three lines in order
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Enter long: ADX > 25 (strong trend) + bullish alignment
            if adx_aligned[i] > 25 and bullish_alignment:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (strong trend) + bearish alignment
            elif adx_aligned[i] > 25 and bearish_alignment:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (weakening trend) or bearish alignment
            if adx_aligned[i] < 20 or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (weakening trend) or bullish alignment
            if adx_aligned[i] < 20 or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals