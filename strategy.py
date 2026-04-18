#!/usr/bin/env python3
"""
Hypothesis: 6h ADX(14) trend strength + 1w ADX(14) regime filter + volume confirmation.
- Trend: ADX(14) > 25 on 6h + price > 20 EMA on 6h (long) or < 20 EMA (short)
- Regime: 1w ADX(14) > 20 (trending market) to avoid ranging markets
- Volume: current volume > 1.5 * 20-period average for confirmation
Designed for 12-37 trades/year (50-150 total) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original indices
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = np.full(len(high), np.nan)
    dm_plus_smooth = np.full(len(high), np.nan)
    dm_minus_smooth = np.full(len(high), np.nan)
    
    if len(high) >= period:
        atr[period-1] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full(len(high), np.nan)
    di_minus = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.full(len(high), np.nan)
    for i in range(2*period-1, len(high)):
        valid_dx = dx[period:i+1]
        if len(valid_dx) >= period:
            adx[i] = np.nanmean(valid_dx[-period:])
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for ADX and EMA
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 6h
    adx_14_6h = calculate_adx(high_6h, low_6h, close_6h, 14)
    # Calculate EMA(20) on 6h
    ema_20_6h = calculate_ema(close_6h, 20)
    
    # Calculate ADX(14) on 1w for regime filter
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align to main timeframe
    adx_14_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_14_6h)
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need ADX and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_6h_aligned[i]) or np.isnan(ema_20_6h_aligned[i]) or 
            np.isnan(adx_14_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: strong uptrend (ADX>25), price above EMA, trending regime, volume
            if (adx_14_6h_aligned[i] > 25 and 
                close[i] > ema_20_6h_aligned[i] and 
                adx_14_1w_aligned[i] > 20 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (ADX>25), price below EMA, trending regime, volume
            elif (adx_14_6h_aligned[i] > 25 and 
                  close[i] < ema_20_6h_aligned[i] and 
                  adx_14_1w_aligned[i] > 20 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens (ADX<20) or price crosses below EMA
            if adx_14_6h_aligned[i] < 20 or close[i] <= ema_20_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens (ADX<20) or price crosses above EMA
            if adx_14_6h_aligned[i] < 20 or close[i] >= ema_20_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX14_EMA20_1wADX_Volume"
timeframe = "6h"
leverage = 1.0