#!/usr/bin/env python3

"""
Hypothesis: 6-hour Trend-Following Strategy with Daily ADX and Volume Confirmation.
Uses daily ADX (>25) to identify trending markets, 6-hour EMA crossover (34/89) for entry,
and volume spike (>1.5x 20-period average) for confirmation. Exits when ADX weakens (<20)
or EMA crossover reverses. Designed for low trade frequency (15-25 trades/year) to work in
both bull and bear markets by only trading strong trends confirmed by volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA crossover: fast (34) and slow (89)
    ema_fast = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slow = pd.Series(close).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and volume conditions
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Enter long: EMA fast > slow, strong trend, volume spike
            if ema_fast[i] > ema_slow[i] and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: EMA fast < slow, strong trend, volume spike
            elif ema_fast[i] < ema_slow[i] and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend weakens or EMA crossover reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA fast < slow OR trend weakens
                if ema_fast[i] < ema_slow[i] or weak_trend:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA fast > slow OR trend weakens
                if ema_fast[i] > ema_slow[i] or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_EMA_Crossover_Volume"
timeframe = "6h"
leverage = 1.0