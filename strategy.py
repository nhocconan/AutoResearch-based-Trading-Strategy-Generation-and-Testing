#!/usr/bin/env python3
"""
6h Elder Ray Power with 12h ADX Trend Filter.
Long when Bear Power turns positive in strong uptrend (ADX>25).
Short when Bull Power turns negative in strong downtrend (ADX>25).
Exit when power signals reverse or ADX weakens (<20).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_12h_adx_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12H EMA13 TREND (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    twelve_h_close = df_12h['close'].values
    twelve_h_ema13 = pd.Series(twelve_h_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    twelve_h_ema13_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema13)
    
    # === 12H ADX (HTF) ===
    twelve_h_high = df_12h['high'].values
    twelve_h_low = df_12h['low'].values
    twelve_h_close_adx = df_12h['close'].values
    
    # True Range
    tr1 = twelve_h_high[1:] - twelve_h_low[1:]
    tr2 = np.abs(twelve_h_high[1:] - twelve_h_close_adx[:-1])
    tr3 = np.abs(twelve_h_low[1:] - twelve_h_close_adx[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((twelve_h_high[1:] - twelve_h_high[:-1]) > (twelve_h_low[:-1] - twelve_h_low[1:]),
                       np.maximum(twelve_h_high[1:] - twelve_h_high[:-1], 0), 0)
    dm_minus = np.where((twelve_h_low[:-1] - twelve_h_low[1:]) > (twelve_h_high[1:] - twelve_h_high[:-1]),
                        np.maximum(twelve_h_low[:-1] - twelve_h_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === ELDER RAY POWER (6H) ===
    # EMA13 for 6h
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        if (np.isnan(twelve_h_ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Trend and strength from 12h
        uptrend = close[i] > twelve_h_ema13_aligned[i]
        downtrend = close[i] < twelve_h_ema13_aligned[i]
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive OR trend weakens
            if bear_power[i] > 0 or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative OR trend weakens
            if bull_power[i] < 0 or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend
            if not strong_trend:
                signals[i] = 0.0
                continue
            
            # Entry: Power signals with trend alignment
            if bear_power[i] < 0 and bull_power[i] > 0 and uptrend:
                # Bull Power positive, Bear Power negative in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif bull_power[i] > 0 and bear_power[i] < 0 and downtrend:
                # Bull Power positive, Bear Power negative in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals