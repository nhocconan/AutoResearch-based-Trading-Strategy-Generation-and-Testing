#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot reversal with 1d ADX trend filter and volume confirmation
# Long when price touches Camarilla S3 level with 1d ADX > 20 and volume > 1.2x average
# Short when price touches Camarilla R3 level with 1d ADX > 20 and volume > 1.2x average
# Exit when price crosses the Camarilla pivot point (mean of high, low, close from 1d)
# Combines mean reversion at key levels with trend strength and volume to avoid false signals
# Designed for high-conviction trades in both trending and ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25

name = "4h_Camarilla_Reversal_1dADX20_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        # Wilder's smoothing
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    
    if len(dx) >= tr_period:
        adx[2*tr_period-2] = np.nanmean(dx[tr_period-1:2*tr_period-1])
        for i in range(2*tr_period-1, len(dx)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d candle
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla levels: H/L/C from previous day
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    camarilla_s3 = camarilla_pivot - 1.1 * range_hl / 6
    camarilla_r3 = camarilla_pivot + 1.1 * range_hl / 6
    
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.2 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches or goes below S3 with trend filter and volume
            if (low[i] <= camarilla_s3_aligned[i] and 
                adx_aligned[i] > 20 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches or goes above R3 with trend filter and volume
            elif (high[i] >= camarilla_r3_aligned[i] and 
                  adx_aligned[i] > 20 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above pivot point
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below pivot point
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals