#!/usr/bin/env python3
"""
1h_Position_Sizing_With_4h_Trend_Filter
Hypothesis: Use 4h ADX to determine trend strength and direction, then take 1h pullbacks to 20 EMA in the trend direction with volume confirmation. This combines trend filtering (4h ADX > 25) with mean-reversion entries (1h price near 20 EMA) to capture swings in both bull and bear markets. Volume > 1.5x average confirms institutional interest. Targets 15-30 trades/year by requiring strong trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth using Wilder's smoothing
    tr_period = np.zeros_like(tr)
    plus_dm_period = np.zeros_like(plus_dm)
    minus_dm_period = np.zeros_like(minus_dm)
    
    tr_period[0] = tr[0]
    plus_dm_period[0] = plus_dm[0]
    minus_dm_period[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
        minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
    
    # Calculate DX
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    return adx

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    ema[0] = close[0]
    multiplier = 2 / (period + 1)
    for i in range(1, len(close)):
        ema[i] = close[i] * multiplier + ema[i-1] * (1 - multiplier)
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for trend filtering
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend strength and direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Calculate 4h EMA20 for trend direction
    ema20_4h = calculate_ema(close_4h, 20)
    
    # Align 4h indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1h EMA20 for pullback entries
    close_1h = prices['close'].values
    ema20_1h = calculate_ema(close_1h, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(adx_4h_aligned[i]) or np.isnan(ema20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        adx = adx_4h_aligned[i]
        ema20_4h = ema20_4h_aligned[i]
        ema20_1h = ema20_1h[i]
        close_price = close_1h[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend direction from 4h EMA20
        # Need previous 4h EMA value to determine slope
        if i >= 1:
            ema20_4h_prev = ema20_4h_aligned[i-1]
            uptrend = ema20_4h > ema20_4h_prev
            downtrend = ema20_4h < ema20_4h_prev
        else:
            uptrend = False
            downtrend = False
        
        # Distance from 1h EMA20 as percentage
        if ema20_1h != 0:
            price_vs_ema = (close_price - ema20_1h) / ema20_1h * 100
        else:
            price_vs_ema = 0
        
        # Entry conditions
        if position == 0:
            # Long: uptrend (ADX > 25) + pullback to EMA20 + volume
            if adx > 25 and uptrend and abs(price_vs_ema) < 0.5 and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (ADX > 25) + pullback to EMA20 + volume
            elif adx > 25 and downtrend and abs(price_vs_ema) < 0.5 and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks down or price moves too far from EMA
            if not uptrend or adx < 20 or abs(price_vs_ema) > 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend breaks up or price moves too far from EMA
            if not downtrend or adx < 20 or abs(price_vs_ema) > 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Position_Sizing_With_4h_Trend_Filter"
timeframe = "1h"
leverage = 1.0