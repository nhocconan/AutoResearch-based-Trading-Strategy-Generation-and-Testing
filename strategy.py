#!/usr/bin/env python3
"""
Hypothesis: 4h price crossing 12h EMA(34) with volume confirmation and 1d ADX(14) trend filter.
In trending markets (ADX>25): follow EMA direction (long above, short below).
In ranging markets (ADX<=25): fade extremes (short above, long below).
Volume confirms institutional participation.
Designed for ~20-30 trades/year to minimize fee drag.
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
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_ma = np.full(len(tr), np.nan)
    dm_plus_ma = np.full(len(dm_plus), np.nan)
    dm_minus_ma = np.full(len(dm_minus), np.nan)
    
    tr_ma[period-1] = np.sum(tr[:period])
    dm_plus_ma[period-1] = np.sum(dm_plus[:period])
    dm_minus_ma[period-1] = np.sum(dm_minus[:period])
    
    for i in range(period, len(tr)):
        tr_ma[i] = tr_ma[i-1] - (tr_ma[i-1] / period) + tr[i]
        dm_plus_ma[i] = dm_plus_ma[i-1] - (dm_plus_ma[i-1] / period) + dm_plus[i]
        dm_minus_ma[i] = dm_minus_ma[i-1] - (dm_minus_ma[i-1] / period) + dm_minus[i]
    
    # Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period-1, len(tr)):
        if tr_ma[i] != 0:
            plus_di[i] = 100 * dm_plus_ma[i] / tr_ma[i]
            minus_di[i] = 100 * dm_minus_ma[i] / tr_ma[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # DX and ADX
    dx = np.full(len(tr), np.nan)
    for i in range(period-1, len(tr)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx = np.full(len(dx), np.nan)
    adx[2*period-2] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period-1, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Get 1d data for ADX(14)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Calculate ADX(14) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(adx_14_1d_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Trending market (ADX > 25): follow EMA direction
            if adx_14_1d_4h[i] > 25:
                if close[i] > ema_34_12h_4h[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema_34_12h_4h[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX <= 25): fade extremes
            else:
                if close[i] > ema_34_12h_4h[i] and vol_confirmed:
                    signals[i] = -0.25  # Short at resistance
                    position = -1
                elif close[i] < ema_34_12h_4h[i] and vol_confirmed:
                    signals[i] = 0.25   # Long at support
                    position = 1
        
        elif position == 1:
            # Long exit: trend change or mean reversion signal
            if (adx_14_1d_4h[i] > 25 and close[i] <= ema_34_12h_4h[i]) or \
               (adx_14_1d_4h[i] <= 25 and close[i] >= ema_34_12h_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or mean reversion signal
            if (adx_14_1d_4h[i] > 25 and close[i] >= ema_34_12h_4h[i]) or \
               (adx_14_1d_4h[i] <= 25 and close[i] <= ema_34_12h_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34_12h_ADX14_Volume"
timeframe = "4h"
leverage = 1.0