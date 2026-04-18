#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 with volume confirmation and 1d ADX trend filter captures breakouts in trending markets while avoiding false signals in ranging markets. Uses 1d ATR for stop loss via signal reversal. Designed for fewer trades (target 12-37/year) to minimize fee drag on 12h timeframe.
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for previous 12h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rango_12h = high_12h - low_12h
    r1_12h = close_12h + rango_12h * 1.1 / 12
    s1_12h = close_12h - rango_12h * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (previous bar values)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    adx_period = 14
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr_1d = np.full_like(close_1d, np.nan)
    dm_plus_smooth = np.full_like(close_1d, np.nan)
    dm_minus_smooth = np.full_like(close_1d, np.nan)
    
    if len(tr) >= adx_period:
        # Initial values
        atr_1d[adx_period] = np.nansum(tr[1:adx_period+1])
        dm_plus_smooth[adx_period] = np.nansum(dm_plus[1:adx_period+1])
        dm_minus_smooth[adx_period] = np.nansum(dm_minus[1:adx_period+1])
        
        # Wilder smoothing
        for i in range(adx_period + 1, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (adx_period - 1) + tr[i]) / adx_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (adx_period - 1) + dm_plus[i]) / adx_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (adx_period - 1) + dm_minus[i]) / adx_period
    
    # DI and DX
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX smoothing
    adx_1d = np.full_like(close_1d, np.nan)
    if len(dx) >= adx_period * 2:
        adx_1d[adx_period*2-1] = np.nanmean(dx[adx_period:adx_period*2])
        for i in range(adx_period*2, len(close_1d)):
            adx_1d[i] = (adx_1d[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1  # Need at least one 12h bar for Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and trending:
            # Long: price breaks above R1 with volume
            if close[i] > r1_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < s1_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ADX weakens (< 20)
            if close[i] < s1_12h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ADX weakens (< 20)
            if close[i] > r1_12h_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0