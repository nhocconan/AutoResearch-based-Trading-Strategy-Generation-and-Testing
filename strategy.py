#!/usr/bin/env python3
"""
1h ADX + Volume Spike with 4h Trend Filter
Hypothesis: Strong directional moves with volume confirmation and higher timeframe trend alignment capture trending moves while avoiding whipsaws. Works in both bull and bear markets by filtering entries with 4h ADX trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[period] = np.sum(tr[:period])
    dm_plus_period[period] = np.sum(dm_plus[:period])
    dm_minus_period[period] = np.sum(dm_minus[:period])
    
    for i in range(period + 1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Directional Indicators
    plus_di = 100 * dm_plus_period / tr_period
    minus_di = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = np.zeros_like(tr)
    dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = np.zeros_like(tr)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h, _, _ = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1h ADX for entry confirmation
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(adx_1h[i]) or 
            np.isnan(plus_di_1h[i]) or np.isnan(minus_di_1h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Require 4h trend strength > 25
        if adx_4h_aligned[i] < 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: +DI > -DI and volume spike
            if plus_di_1h[i] > minus_di_1h[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: -DI > +DI and volume spike
            elif minus_di_1h[i] > plus_di_1h[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: -DI crosses above +DI or volume spike ends
            if minus_di_1h[i] > plus_di_1h[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: +DI crosses above -DI or volume spike ends
            if plus_di_1h[i] > minus_di_1h[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ADX_VolumeSpike_4hTrendFilter"
timeframe = "1h"
leverage = 1.0