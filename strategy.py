#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_1wTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from 1d combined with 1w trend filter and volume spikes capture swing moves in BTC/ETH. Works in bull by buying pullbacks to S1/S2 in uptrend, and in bear by selling rallies to R1/R2 in downtrend. Volume confirmation ensures momentum, reducing false signals. Targets 10-20 trades/year.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R4 = pivot + (range_hl * 1.1 / 2)
    R3 = pivot + (range_hl * 1.1 / 4)
    R2 = pivot + (range_hl * 1.1 / 6)
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (already daily, so just forward fill)
    R1_1d = np.where(np.isnan(R1), np.nan, R1)
    R2_1d = np.where(np.isnan(R2), np.nan, R2)
    S1_1d = np.where(np.isnan(S1), np.nan, S1)
    S2_1d = np.where(np.isnan(S2), np.nan, S2)
    
    # Forward fill to handle NaN from shift
    def ffill(arr):
        mask = np.isnan(arr)
        if not np.any(mask):
            return arr
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]
    
    R1_ff = ffill(R1_1d)
    R2_ff = ffill(R2_1d)
    S1_ff = ffill(S1_1d)
    S2_ff = ffill(S2_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: >2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(R1_ff[i]) or np.isnan(R2_ff[i]) or
            np.isnan(S1_ff[i]) or np.isnan(S2_ff[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Price near Camarilla levels (within 0.5% tolerance)
        tol = 0.005
        near_R1 = abs(close[i] - R1_ff[i]) / R1_ff[i] < tol
        near_R2 = abs(close[i] - R2_ff[i]) / R2_ff[i] < tol
        near_S1 = abs(close[i] - S1_ff[i]) / S1_ff[i] < tol
        near_S2 = abs(close[i] - S2_ff[i]) / S2_ff[i] < tol
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: 
        # Long: price near S1/S2 in uptrend with volume
        # Short: price near R1/R2 in downtrend with volume
        long_entry = vol_confirm and uptrend and (near_S1 or near_S2)
        short_entry = vol_confirm and downtrend and (near_R1 or near_R2)
        
        # Exit logic: opposite signal or trend change
        long_exit = (near_R1 or near_R2) or (not uptrend)
        short_exit = (near_S1 or near_S2) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_Pivot_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0