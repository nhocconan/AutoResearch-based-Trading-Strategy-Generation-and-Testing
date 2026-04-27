#!/usr/bin/env python3
"""
4h_Trix_Signal_Line_Crossover
Momentum strategy using TRIX and its signal line on 4h timeframe.
Long when TRIX crosses above its signal line with volume confirmation.
Short when TRIX crosses below its signal line with volume confirmation.
Exit when opposite crossover occurs.
Uses 1d ADX > 20 to filter for trending markets only.
Target: 20-35 trades/year per symbol.
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
    
    # TRIX parameters
    trix_period = 15
    signal_period = 9
    
    # Calculate TRIX: EMA of EMA of EMA of log(close)
    log_close = np.log(close)
    
    # First EMA
    ema1 = np.full(n, np.nan)
    alpha1 = 2 / (trix_period + 1)
    for i in range(trix_period - 1, n):
        if i == trix_period - 1:
            ema1[i] = np.mean(log_close[i - trix_period + 1:i + 1])
        else:
            ema1[i] = log_close[i] * alpha1 + ema1[i - 1] * (1 - alpha1)
    
    # Second EMA
    ema2 = np.full(n, np.nan)
    alpha2 = 2 / (trix_period + 1)
    for i in range(trix_period - 1, n):
        if i == trix_period - 1:
            ema2[i] = np.mean(ema1[i - trix_period + 1:i + 1])
        else:
            ema2[i] = ema1[i] * alpha2 + ema2[i - 1] * (1 - alpha2)
    
    # Third EMA
    ema3 = np.full(n, np.nan)
    alpha3 = 2 / (trix_period + 1)
    for i in range(trix_period - 1, n):
        if i == trix_period - 1:
            ema3[i] = np.mean(ema2[i - trix_period + 1:i + 1])
        else:
            ema3[i] = ema2[i] * alpha3 + ema3[i - 1] * (1 - alpha3)
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = np.full(n, np.nan)
    for i in range(trix_period, n):
        if ema3[i] != 0 and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    
    # Signal line: EMA of TRIX
    trix_signal = np.full(n, np.nan)
    alpha_signal = 2 / (signal_period + 1)
    for i in range(signal_period - 1, n):
        if i == signal_period - 1:
            # Find first non-NaN TRIX value
            start_idx = trix_period
            while start_idx <= i and np.isnan(trix[start_idx]):
                start_idx += 1
            if start_idx <= i:
                valid_trix = trix[start_idx:i+1]
                valid_trix = valid_trix[~np.isnan(valid_trix)]
                if len(valid_trix) > 0:
                    trix_signal[i] = np.mean(valid_trix)
        else:
            if not np.isnan(trix[i]):
                trix_signal[i] = trix[i] * alpha_signal + trix_signal[i - 1] * (1 - alpha_signal)
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    adx_period = 14
    tr = np.full(len(close_1d), np.nan)
    dm_plus = np.full(len(close_1d), np.nan)
    dm_minus = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
        
        dm_plus[i] = max(0, high_1d[i] - high_1d[i-1])
        dm_minus[i] = max(0, low_1d[i-1] - low_1d[i])
        
        # Apply Wilder's smoothing
        if dm_plus[i] < 0: dm_plus[i] = 0
        if dm_minus[i] < 0: dm_minus[i] = 0
    
    # Smooth TR, DM+, DM-
    tr_smooth = np.full(len(close_1d), np.nan)
    dm_plus_smooth = np.full(len(close_1d), np.nan)
    dm_minus_smooth = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= adx_period:
        # Initial values
        tr_smooth[adx_period-1] = np.nansum(tr[1:adx_period+1])
        dm_plus_smooth[adx_period-1] = np.nansum(dm_plus[1:adx_period+1])
        dm_minus_smooth[adx_period-1] = np.nansum(dm_minus[1:adx_period+1])
        
        # Wilder smoothing
        for i in range(adx_period, len(close_1d)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/adx_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/adx_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/adx_period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.full(len(close_1d), np.nan)
    di_minus = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    for i in range(adx_period-1, len(close_1d)):
        if tr_smooth[i] != 0 and not np.isnan(tr_smooth[i]):
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 2*adx_period-1:
        adx[2*adx_period-2] = np.nanmean(dx[adx_period-1:2*adx_period-1])
        for i in range(2*adx_period-1, len(close_1d)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need TRIX, signal line, volume MA, and ADX
    start_idx = max(trix_period + signal_period, vol_period - 1, 2*adx_period-1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # ADX filter: trending market (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume and trend
            if (trix[i] > trix_signal[i] and 
                trix[i-1] <= trix_signal[i-1] and 
                volume_confirmed and trending):
                signals[i] = size
                position = 1
            # Short: TRIX crosses below signal line with volume and trend
            elif (trix[i] < trix_signal[i] and 
                  trix[i-1] >= trix_signal[i-1] and 
                  volume_confirmed and trending):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Trix_Signal_Line_Crossover"
timeframe = "4h"
leverage = 1.0