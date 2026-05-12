#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_4hTrend
Hypothesis: Price breaking above/below 4h Donchian(20) channels with 4h EMA(50) trend filter and volume confirmation (2x average) captures strong trending moves while avoiding false breakouts. Uses 1d ADX>25 as trend regime filter to avoid ranging markets. Designed for 4h timeframe with ~25-40 trades/year per symbol. Works in bull/bear by following 4h trend direction and avoiding low-volatility chop.
"""

name = "4h_Donchian_20_Breakout_Volume_Trend_4hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA(50) trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: >2x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 1d ADX(14) for trend regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe (with 1-bar delay for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Trend regime: ADX > 25 indicates trending market
    trending_regime = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(trending_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 4h EMA50 uptrend + volume spike + trending regime
            if (close[i] > high_roll[i] and 
                close[i] > ema_50[i] and 
                volume_spike[i] and 
                trending_regime[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 4h EMA50 downtrend + volume spike + trending regime
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50[i] and 
                  volume_spike[i] and 
                  trending_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals