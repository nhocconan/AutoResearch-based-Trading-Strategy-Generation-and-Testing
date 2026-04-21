#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_ADX_Trend
Hypothesis: Breakouts above/below 4h Donchian(20) channels with volume confirmation and ADX > 20 trend filter yield robust trend-following trades. Works in bull/bear markets by only taking breakouts in the direction of the trend (ADX > 20). Volume spike (2x 20-period average) confirms breakout strength. Targets 20-50 trades/year by requiring confluence of breakout, volume, and trend. Uses 4h as primary timeframe with 1d ADX for trend strength to avoid lower-timeframe noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth using Wilder's smoothing
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[0] = tr[0]
    dm_plus_period[0] = dm_plus[0]
    dm_minus_period[0] = dm_minus[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Directional Indicators
    plus_di = 100 * dm_plus_period / tr_period
    minus_di = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = np.zeros_like(tr)
    dx[tr_period != 0] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = np.zeros_like(dx)
    if len(dx) >= period:
        adx[period-1] = np.nanmean(dx[:period])  # First ADX is average of first 'period' DX values
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if ADX not ready
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 20 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = prices['volume'].iloc[i] > 2.0 * vol_ma
        else:
            volume_ok = False
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume + strong trend
            if price > donchian_upper[i] and volume_ok and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume + strong trend
            elif price < donchian_lower[i] and volume_ok and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend weakens
            if price < donchian_lower[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend weakens
            if price > donchian_upper[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADX_Trend"
timeframe = "4h"
leverage = 1.0