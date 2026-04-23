#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot R3/S3 Fade with 1d Volume Spike and ADX Regime Filter
- Uses 6h Camarilla pivot levels (R3/S3) for mean reversion entries
- Fades extreme moves: short at R3, long at S3 with volume confirmation (> 1.5x 20-period average)
- 1d ADX > 25 filters for trending markets (avoid fading in strong trends)
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in ranging markets (camarilla fade) and avoids trending markets via ADX filter
"""

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
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    period = 14
    tr_period = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # need 1d ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or 
            i < 5):  # need 5 bars for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels for 6h using previous day's OHLC
        # We need to get the previous daily OHLC from 1d data
        # Find the index of the previous completed 1d bar
        # Since we're on 6h timeframe, we can approximate using rolling window
        if i >= 4:  # at least 4*6h = 24h to get daily OHLC
            # Get the highest high, lowest low, and close from the last 4 bars (approx 1 day)
            day_high = np.max(high[i-4:i])
            day_low = np.min(low[i-4:i])
            day_close = close[i-1]  # previous bar close
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r3 = day_close + range_val * 1.1 / 4
        camarilla_s3 = day_close - range_val * 1.1 / 4
        camarilla_r4 = day_close + range_val * 1.1 / 2
        camarilla_s4 = day_close - range_val * 1.1 / 2
        
        # ADX regime filter: only trade when ADX < 25 (ranging market)
        if adx_aligned[i] >= 25:
            # Strong trend - avoid fading, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price reaches S3 with volume confirmation
            if (low[i] <= camarilla_s3 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price reaches R3 with volume confirmation
            elif (high[i] >= camarilla_r3 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite S3/R3 level or midpoint
            exit_signal = False
            camarilla_mid = (camarilla_r3 + camarilla_s3) / 2
            
            if position == 1:
                # Exit long when price reaches R3 or crosses midpoint
                if high[i] >= camarilla_r3 or close[i] >= camarilla_mid:
                    exit_signal = True
            elif position == -1:
                # Exit short when price reaches S3 or crosses midpoint
                if low[i] <= camarilla_s3 or close[i] <= camarilla_mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_1dADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0