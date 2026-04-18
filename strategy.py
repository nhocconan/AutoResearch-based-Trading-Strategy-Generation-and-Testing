#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter.
- Elder Ray = Bull Power (high - EMA13), Bear Power (low - EMA13)
- Regime: 1d ADX(14) > 25 = trending, < 20 = ranging
- Trending regime: go with Elder Ray (BP > 0 long, BP < 0 short)
- Ranging regime: fade extremes (BP > 0.5*ATR short, BP < -0.5*ATR long)
- Volume filter: volume > 1.2x 20-period average
- Target: 50-150 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(data, period):
    """Calculate Exponential Moving Average."""
    if len(data) < period:
        return np.full(len(data), np.nan)
    ema = np.full(len(data), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d for Elder Ray components
    ema13_1d = calculate_ema(close_1d, 13)
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power_1d = high_1d - ema13_1d  # High - EMA
    bear_power_1d = low_1d - ema13_1d   # Low - EMA
    
    # Calculate ATR on 1d for threshold
    atr14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate ADX on 1d for regime
    adx14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    atr14_1d_6h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    adx14_1d_6h = align_htf_to_ltf(prices, df_1d, adx14_1d)
    
    # Calculate volume moving average (20-period) on 6h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need EMA13, ATR14, ADX14, and vol MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(atr14_1d_6h[i]) or np.isnan(adx14_1d_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Regime detection
        trending = adx14_1d_6h[i] > 25
        ranging = adx14_1d_6h[i] < 20
        
        if position == 0:
            if trending and vol_confirmed:
                # Trending regime: follow Elder Ray power
                if bull_power_6h[i] > 0:
                    signals[i] = 0.25
                    position = 1
                elif bear_power_6h[i] < 0:
                    signals[i] = -0.25
                    position = -1
            elif ranging and vol_confirmed:
                # Ranging regime: fade extremes
                atr_threshold = 0.5 * atr14_1d_6h[i]
                if bear_power_6h[i] < -atr_threshold:  # Oversold: go long
                    signals[i] = 0.25
                    position = 1
                elif bull_power_6h[i] > atr_threshold:  # Overbought: go short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if trending:
                # Exit when bull power fades
                if bull_power_6h[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit when bear power becomes positive (mean reversion complete)
                if bear_power_6h[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if trending:
                # Exit when bear power recovers
                if bear_power_6h[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit when bull power becomes negative (mean reversion complete)
                if bull_power_6h[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0