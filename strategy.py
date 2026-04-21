#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter_V2
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) on 12h timeframe provides adaptive trend direction. 
Enter long when price crosses above KAMA with volume confirmation (1.5x 20-period average) and ADX > 20. 
Enter short when price crosses below KAMA with same conditions. 
Exit when price crosses back below/above KAMA or ADX drops below 15 (weakening trend). 
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. 
Targets 15-35 trades/year by requiring both volume and trend confirmation. 
Works in bull/bear markets by following adaptive trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, prepend=close[0])).sum()
    # Handle first er_period values
    for i in range(len(change)):
        if i < er_period:
            change[i] = np.abs(close[i] - close[0])
            volatility = np.sum(np.abs(np.diff(close[:i+1])))
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

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
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])  # First ADX value
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate KAMA on 12h price
    kama = calculate_kama(prices['close'].values, er_period=10, fast_ema=2, slow_ema=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ADX not ready
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        kama_val = kama[i]
        prev_kama = kama[i-1]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: price crosses above KAMA + volume confirmation + trending market
            if price > kama_val and price <= prev_kama and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA + volume confirmation + trending market
            elif price < kama_val and price >= prev_kama and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or ADX drops below 15 (weakening trend)
            if price < kama_val or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or ADX drops below 15 (weakening trend)
            if price > kama_val or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Filter_V2"
timeframe = "12h"
leverage = 1.0