#!/usr/bin/env python3
"""
4h_VolumeWeighted_KAMA_Reversal_v1
Hypothesis: Combine KAMA (Kaufman Adaptive Moving Average) with volume-weighted price to identify trend exhaustion points. In both bull and bear markets, sharp price moves with declining volume often precede reversals. Volume-weighted KAMA adapts faster during high-volume moves and slower during low-volume consolidations, providing early reversal signals. Uses 1d ADX for regime filtering to avoid whipsaws in low-trend environments. Designed for low trade frequency (<25/year) to minimize fee drag.
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
    
    # Calculate Volume-Weighted Close (VWC)
    vwc = (close * volume) / (volume + 1e-10)
    
    # Calculate Kaufman Adaptive Moving Average (KAMA) on VWC
    # KAMA parameters: ER period=10, Fast SC=2, Slow SC=30
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(vwc, prepend=vwc[0]))
    direction = np.abs(np.subtract(vwc, np.roll(vwc, er_period)))
    volatility = np.cumsum(change)
    volatility = np.where(np.arange(len(vwc)) < er_period, 
                         np.full_like(volatility, np.nan), 
                         volatility - np.roll(volatility, er_period))
    er = np.where(volatility != 0, direction / volatility, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(vwc, np.nan)
    kama[0] = vwc[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (vwc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1d ADX for regime filtering (trend strength)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = np.subtract(high_1d, low_1d)
    tr2 = np.subtract(np.abs(high_1d), np.abs(np.roll(close_1d, 1)))
    tr3 = np.subtract(np.abs(low_1d), np.abs(np.roll(close_1d, 1)))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.where(np.arange(len(tr)) == 0, high_1d[0] - low_1d[0], tr)
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, +DM, -DM
    atr_period = 14
    atr = np.full_like(high_1d, np.nan)
    plus_dm_smooth = np.full_like(high_1d, np.nan)
    minus_dm_smooth = np.full_like(high_1d, np.nan)
    
    # Initial values
    if len(high_1d) >= atr_period:
        atr[atr_period-1] = np.nansum(tr[:atr_period])
        plus_dm_smooth[atr_period-1] = np.nansum(plus_dm[:atr_period])
        minus_dm_smooth[atr_period-1] = np.nansum(minus_dm[:atr_period])
    
    # Wilder's smoothing
    for i in range(atr_period, len(high_1d)):
        atr[i] = atr[i-1] - (atr[i-1] / atr_period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / atr_period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / atr_period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Calculate ADX (smoothed DX)
    adx_period = 14
    adx = np.full_like(dx, np.nan)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.nansum(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / adx_period) + dx[i]
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_series = pd.Series(volume)
    vol_ma[20:] = vol_series.rolling(window=20, min_periods=20).mean().values[20:]
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_period, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_val < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume spike
            if price > kama_val and close[i-1] <= kama_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume spike
            elif price < kama_val and close[i-1] >= kama_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or volume drops significantly
            if price < kama_val or (volume[i] < 0.5 * vol_ma[i] and not np.isnan(vol_ma[i])):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or volume drops significantly
            if price > kama_val or (volume[i] < 0.5 * vol_ma[i] and not np.isnan(vol_ma[i])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_VolumeWeighted_KAMA_Reversal_v1"
timeframe = "4h"
leverage = 1.0