#!/usr/bin/env python3
# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1d ADX trend filter
# Long when price breaks above Camarilla R4 level with 1d ADX > 20 and volume > 2x average
# Short when price breaks below Camarilla S4 level with 1d ADX > 20 and volume > 2x average
# Exit when price returns to Camarilla pivot (midpoint)
# Camarilla levels provide institutional support/resistance, ADX filters trend strength, volume confirms breakout validity
# Designed for low-frequency, high-conviction trades on 12h timeframe
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Camarilla_R4_S4_Breakout_1dADX20_Volume"
timeframe = "12h"
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
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels from previous day
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_ = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4 = close_1d[:-1] + range_ * 1.1 / 2
    s4 = close_1d[:-1] - range_ * 1.1 / 2
    pivot_level = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3  # Same as pivot
    
    # Calculate 1d ADX for trend filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    
    if len(dx) >= tr_period:
        adx[2*tr_period-2] = np.nanmean(dx[tr_period-1:2*tr_period-1])
        for i in range(2*tr_period-1, len(dx)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Align 1d data to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4, additional_delay_bars=1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4, additional_delay_bars=1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R4 with trend and volume
            if (close[i] > r4_aligned[i] and 
                adx_aligned[i] > 20 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S4 with trend and volume
            elif (close[i] < s4_aligned[i] and 
                  adx_aligned[i] > 20 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Camarilla pivot
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Camarilla pivot
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals