#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1d Volume Spike and ADX Trend Filter
Hypothesis: Price reversals at Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts)
on 12h timeframe, filtered by 1d volume spikes and ADX > 25 for trend strength,
captures mean-reversion bounces in ranging markets and avoids false signals in strong trends.
Works in bull via S3 bounces, in bear via R3 rejections. Volume and ADX filters reduce whipsaws.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX(14) for trend filter
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_14 = adx
    
    # Align ADX to 12h
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 1d volume filter: current volume > 2.0x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's range
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    range_ = prev_high - prev_low
    
    # Camarilla levels
    S3 = prev_close - (range_ * 1.1 / 6)
    S4 = prev_close - (range_ * 1.1 / 2)
    R3 = prev_close + (range_ * 1.1 / 6)
    R4 = prev_close + (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 12h
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S4 (strong support) or ADX > 30 (strong trend)
            if close[i] <= S4_aligned[i] or adx_14_aligned[i] > 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 (strong resistance) or ADX > 30 (strong trend)
            if close[i] >= R4_aligned[i] or adx_14_aligned[i] > 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: only trade when ADX < 25 (not strong trend)
            if adx_14_aligned[i] >= 25:
                signals[i] = 0.0
                continue
            
            # Volume filter: require volume spike
            if not vol_spike_1d_aligned[i]:
                signals[i] = 0.0
                continue
            
            # Long: price touches or goes below S3 (strong support) with volume spike
            if close[i] <= S3_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches or goes above R3 (strong resistance) with volume spike
            elif close[i] >= R3_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals