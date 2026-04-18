#!/usr/bin/env python3
"""
Hypothesis: 12h 12-period EMA trend with 1d volume spike filter and 1d ADX regime filter.
- Long: Price > EMA(12) on 12h, volume > 2.0x 20-period average on 1d, ADX > 25 on 1d
- Short: Price < EMA(12) on 12h, volume > 2.0x 20-period average on 1d, ADX > 25 on 1d
- Exit: Price crosses back below/above EMA(12) or volume drops below 1.5x average
- Uses EMA for trend, volume spike for momentum confirmation, ADX to avoid ranging markets.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    ema = np.full(len(close), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    
    for i in range(period, len(close)):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    if len(dm_plus) >= period:
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        for i in range(period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.full(len(plus_di), np.nan)
    for i in range(period, len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA (12-period) on 12h
    ema_12 = calculate_ema(close, 12)
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Calculate ADX (14-period) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_14_1d_12h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need EMA, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12[i]) or np.isnan(vol_ma_1d_12h[i]) or 
            np.isnan(adx_14_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        # Get the 1d volume for current 12h bar (use aligned volume data)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike = vol_1d_aligned[i] > 2.0 * vol_ma_1d_12h[i]
        
        if position == 0:
            # Long: price above EMA, volume spike, ADX > 25
            if close[i] > ema_12[i] and vol_spike and adx_14_1d_12h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA, volume spike, ADX > 25
            elif close[i] < ema_12[i] and vol_spike and adx_14_1d_12h[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA or volume drops below 1.5x average
            vol_exit = vol_1d_aligned[i] < 1.5 * vol_ma_1d_12h[i]
            if close[i] <= ema_12[i] or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA or volume drops below 1.5x average
            vol_exit = vol_1d_aligned[i] < 1.5 * vol_ma_1d_12h[i]
            if close[i] >= ema_12[i] or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA12_VolumeSpike_ADX14"
timeframe = "12h"
leverage = 1.0