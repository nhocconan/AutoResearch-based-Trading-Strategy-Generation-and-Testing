#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Elder Ray + ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# ADX > 25 confirms trend presence. Enter when Bull/Bear Power aligns with ADX trend
# Exit when power weakens or ADX drops below 20. Designed for 15-30 trades/year.
# Works in both bull/bear by adapting to trend direction via ADX filter.

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on daily close
    close_1d = df_1d['close'].values
    ema13 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 13:
        alpha = 2 / (13 + 1)
        ema13[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13[i] = alpha * close_1d[i] + (1 - alpha) * ema13[i-1]
    
    # Calculate Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13  # High - EMA13
    bear_power = ema13 - low_1d   # EMA13 - Low
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + arr[i]
        return smoothed
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth(dx, 14)
    
    # Align indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Low volatility filter: ADX < 20 (range market)
        weak_trend = adx_aligned[i] < 20
        
        # Elder Ray signals
        bullish = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        bearish = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (bull_power_aligned[i] <= 0 or weak_trend))
        exit_short = (position == -1 and 
                     (bear_power_aligned[i] <= 0 or weak_trend))
        
        # Entry conditions
        if strong_trend and bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_trend and bearish and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals