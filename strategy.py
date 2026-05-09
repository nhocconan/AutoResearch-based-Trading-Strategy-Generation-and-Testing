#!/usr/bin/env python3
# 2025-06-22 | 12h_TRIX_VolumeSpike_Regime_v2
# Hypothesis: TRIX (triple exponential moving average) crossover with volume spike confirmation and ADX trend filter.
# TRIX > 0 indicates bullish momentum, TRIX < 0 indicates bearish momentum.
# Volume spike (>2x 24-period average) confirms momentum strength.
# ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions.
# Designed for 12h timeframe to achieve low trade frequency (12-37/year) and minimize fee drag.
# Works in both bull and bear markets by following momentum with trend filter.

name = "12h_TRIX_VolumeSpike_Regime_v2"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for TRIX calculation (using daily close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of % change
    # Step 1: Calculate % change
    pct_change = np.full_like(close_1d, np.nan)
    pct_change[1:] = (close_1d[1:] - close_1d[:-1]) / close_1d[:-1]
    
    # Step 2: First EMA of % change
    ema1 = np.full_like(close_1d, np.nan)
    if len(pct_change) >= 15:
        ema1[14] = np.nanmean(pct_change[1:15])  # Skip first NaN
        for i in range(15, len(pct_change)):
            if not np.isnan(pct_change[i]):
                ema1[i] = (pct_change[i] * 2 / (15 + 1)) + (ema1[i-1] * (1 - 2 / (15 + 1)))
            else:
                ema1[i] = ema1[i-1]
    
    # Step 3: Second EMA of first EMA
    ema2 = np.full_like(close_1d, np.nan)
    if len(ema1) >= 15:
        valid_start = np.where(~np.isnan(ema1))[0]
        if len(valid_start) > 0:
            start_idx = valid_start[0]
            if len(ema1[start_idx:]) >= 15:
                ema2[start_idx + 14] = np.nanmean(ema1[start_idx:start_idx+15])
                for i in range(start_idx + 15, len(ema1)):
                    if not np.isnan(ema1[i]):
                        ema2[i] = (ema1[i] * 2 / (15 + 1)) + (ema2[i-1] * (1 - 2 / (15 + 1)))
                    else:
                        ema2[i] = ema2[i-1]
    
    # Step 4: Third EMA of second EMA
    ema3 = np.full_like(close_1d, np.nan)
    if len(ema2) >= 15:
        valid_start = np.where(~np.isnan(ema2))[0]
        if len(valid_start) > 0:
            start_idx = valid_start[0]
            if len(ema2[start_idx:]) >= 15:
                ema3[start_idx + 14] = np.nanmean(ema2[start_idx:start_idx+15])
                for i in range(start_idx + 15, len(ema2)):
                    if not np.isnan(ema2[i]):
                        ema3[i] = (ema2[i] * 2 / (15 + 1)) + (ema3[i-1] * (1 - 2 / (15 + 1)))
                    else:
                        ema3[i] = ema3[i-1]
    
    # TRIX = 100 * (third EMA - previous third EMA) / previous third EMA
    trix = np.full_like(close_1d, np.nan)
    valid_idx = np.where(~np.isnan(ema3))[0]
    if len(valid_idx) > 1:
        for i in range(1, len(valid_idx)):
            idx = valid_idx[i]
            prev_idx = valid_idx[i-1]
            if ema3[prev_idx] != 0:
                trix[idx] = 100 * (ema3[idx] - ema3[prev_idx]) / ema3[prev_idx]
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate ADX for trend strength (using daily data)
    # ADX calculation requires high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            smoothed[period-1] = np.nanmean(data[1:period])  # Skip first NaN in TR
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    smoothed[i] = (data[i] * (1/period)) + (smoothed[i-1] * (1 - 1/period))
                else:
                    smoothed[i] = smoothed[i-1]
        return smoothed
    
    atr = wilder_smooth(tr, 14)
    plus_di_smoothed = wilder_smooth(plus_dm, 14)
    minus_di_smoothed = wilder_smooth(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.full_like(close_1d, np.nan)
    minus_di = np.full_like(close_1d, np.nan)
    valid = (~np.isnan(atr)) & (atr != 0)
    plus_di[valid] = 100 * plus_di_smoothed[valid] / atr[valid]
    minus_di[valid] = 100 * minus_di_smoothed[valid] / atr[valid]
    
    # Calculate DX and ADX
    dx = np.full_like(close_1d, np.nan)
    di_sum = plus_di + minus_di
    valid_dx = (~np.isnan(plus_di)) & (~np.isnan(minus_di)) & (di_sum != 0)
    dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / di_sum[valid_dx]
    
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter: current volume / 24-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Ensure TRIX, ADX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX > 0 (bullish momentum) AND ADX > 25 (trending) AND volume spike
            if (trix_aligned[i] > 0 and 
                adx_aligned[i] > 25 and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX < 0 (bearish momentum) AND ADX > 25 (trending) AND volume spike
            elif (trix_aligned[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX <= 0 (momentum fade) OR ADX <= 20 (trend weakening)
            if trix_aligned[i] <= 0 or adx_aligned[i] <= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX >= 0 (momentum fade) OR ADX <= 20 (trend weakening)
            if trix_aligned[i] >= 0 or adx_aligned[i] <= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals