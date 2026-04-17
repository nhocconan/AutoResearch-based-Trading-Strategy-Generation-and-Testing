#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian upper and lower bands
    donch_upper_1d = np.full_like(high_1d, np.nan)
    donch_lower_1d = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            donch_upper_1d[i] = np.max(high_1d[i-19:i+1])
            donch_lower_1d[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donch_upper_1d[i] = np.max(high_1d[max(0, i-9):i+1])
            donch_lower_1d[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            donch_upper_1d[i] = high_1d[0]
            donch_lower_1d[i] = low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR using Wilder's smoothing
    atr_1d = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_1d[i] = tr[i] if not np.isnan(tr[i]) else 0
            else:
                prev_atr = atr_1d[i-1]
                if np.isnan(prev_atr):
                    atr_1d[i] = np.nanmean(tr[1:i+1]) if np.sum(~np.isnan(tr[1:i+1])) > 0 else 0
                else:
                    if np.isnan(tr[i]):
                        atr_1d[i] = prev_atr
                    else:
                        atr_1d[i] = (prev_atr * (period-1) + tr[i]) / period
        else:
            if np.isnan(tr[i]):
                atr_1d[i] = atr_1d[i-1]
            else:
                atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # === 1d Weekly Pivot Points (using prior week's data) ===
    # We'll approximate weekly pivot using daily data (more stable)
    # Pivot = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 0:  # Use current day's data for pivot
            pivot_1d[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            r1_1d[i] = 2 * pivot_1d[i] - low_1d[i]
            s1_1d[i] = 2 * pivot_1d[i] - high_1d[i]
        else:
            pivot_1d[i] = close_1d[0]
            r1_1d[i] = close_1d[0]
            s1_1d[i] = close_1d[0]
    
    # === Align indicators to 6h timeframe ===
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 6h Volume confirmation ===
    # Calculate 20-period average volume on 6h timeframe
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Breakout logic: price breaks Donchian channel with volume
        if position == 0:
            # Long breakout: price > Donchian upper + volume confirmation
            if close[i] > donch_upper_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short breakout: price < Donchian lower + volume confirmation
            elif close[i] < donch_lower_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility filter
        elif position == 1:
            # Exit long: price < Donchian lower OR volatility too low
            if close[i] < donch_lower_aligned[i] or (atr_1d_aligned[i] < np.nanpercentile(atr_1d_aligned[max(0, i-49):i+1], 20) if i >= 49 else False):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian upper OR volatility too low
            if close[i] > donch_upper_aligned[i] or (atr_1d_aligned[i] < np.nanpercentile(atr_1d_aligned[max(0, i-49):i+1], 20) if i >= 49 else False):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0