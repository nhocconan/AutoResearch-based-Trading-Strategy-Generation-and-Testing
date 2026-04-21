#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_Volume_ADXFilter
Hypothesis: Breakouts above daily R1 or below daily S1 on 4h timeframe with volume confirmation (2x average) and ADX > 25 (trending market) yield high-probability trades. Uses ADX to filter for trending conditions only, avoiding whipsaws in ranging markets. Targets 20-50 trades/year by requiring both volume and trend strength confirmation. Works in bull/bear markets by only taking breakouts in direction of trend (ADX confirms trend strength regardless of direction).
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
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[:period])  # First ADX value
    
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot calculation and ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if ADX not ready
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Camarilla levels from previous day's OHLC
        # Align daily OHLC to 4h timeframe
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        
        # Use prior bar's aligned daily values (previous completed day)
        prev_high = high_1d_aligned[i-1]
        prev_low = low_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        
        r1, s1 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + trending market
            if price > r1 and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + trending market
            elif price < s1 and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ADX drops below 20 (losing trend)
            if price < s1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ADX drops below 20 (losing trend)
            if price > r1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0