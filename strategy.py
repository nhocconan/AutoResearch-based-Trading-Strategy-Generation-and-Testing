#!/usr/bin/env python3
"""
4h Donchian Channel Breakout with Volume Spike and ADX Trend Filter
Long: Close breaks above upper band + volume > 2.0 x volume MA(20) + ADX(14) > 25
Short: Close breaks below lower band + volume > 2.0 x volume MA(20) + ADX(14) > 25
Exit: Close crosses back below median line (long) or above median line (short)
Uses 20-period Donchian bands for breakout signals, volume for confirmation, ADX for trend filter
Target: 25-35 trades/year per symbol (100-140 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels on 4h data
    period = 20
    upper_band = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower_band = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # 4h volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) on 4h data
    # Calculate +DI, -DI, DX
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    adx_period = 14
    alpha = 1.0 / adx_period
    
    atr = np.zeros_like(high, dtype=float)
    plus_di = np.zeros_like(high, dtype=float)
    minus_di = np.zeros_like(high, dtype=float)
    dx = np.zeros_like(high, dtype=float)
    adx = np.zeros_like(high, dtype=float)
    
    # Initial values
    atr[adx_period] = np.mean(tr[:adx_period+1])
    plus_dm_smoothed = np.sum(plus_dm[:adx_period+1])
    minus_dm_smoothed = np.sum(minus_dm[:adx_period+1])
    
    for i in range(adx_period + 1, len(high)):
        # Wilder's smoothing
        atr[i] = atr[i-1] - (atr[i-1] / adx_period) + tr[i]
        plus_dm_smoothed = plus_dm_smoothed - (plus_dm_smoothed / adx_period) + plus_dm[i]
        minus_dm_smoothed = minus_dm_smoothed - (minus_dm_smoothed / adx_period) + minus_dm[i]
        
        # Avoid division by zero
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smoothed / atr[i]) * 100
            minus_di[i] = (minus_dm_smoothed / atr[i]) * 100
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0
            dx[i] = 0
    
    # Smoothed ADX
    adx[2*adx_period] = np.mean(dx[adx_period:2*adx_period+1])
    for i in range(2*adx_period + 1, len(high)):
        adx[i] = adx[i-1] - (adx[i-1] / adx_period) + dx[i]
    
    # For indices before sufficient data, set to 0
    adx[:2*adx_period] = 0
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(60, 2*adx_period + 10)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: break above upper band + volume spike + ADX > 25
            if price > upper_band[i] and vol > 2.0 * vol_ma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + volume spike + ADX > 25
            elif price < lower_band[i] and vol > 2.0 * vol_ma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close back below middle band
            if price < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above middle band
            if price > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADX25"
timeframe = "4h"
leverage = 1.0