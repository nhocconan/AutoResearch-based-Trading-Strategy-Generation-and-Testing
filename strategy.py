#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSqueeze
Hypothesis: Combines Camarilla pivot breakouts with volatility squeeze detection and 1d EMA trend filter.
Only trades when volatility is low (squeeze) and price breaks key levels with volume confirmation.
Reduces false breakouts by requiring low volatility environment before entry.
Target: 20-40 total trades over 4 years to minimize fee drag.
Works in both bull and bear markets by using trend filter for direction and squeeze for timing.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r1 = np.zeros(len(close_4h))
    camarilla_s1 = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if high_4h[i] == low_4h[i]:
            camarilla_r1[i] = close_4h[i]
            camarilla_s1[i] = close_4h[i]
        else:
            camarilla_r1[i] = close_4h[i] + (high_4h[i] - low_4h[i]) * 1.1 / 12
            camarilla_s1[i] = close_4h[i] - (high_4h[i] - low_4h[i]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Bollinger Bands for volatility squeeze detection (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    bb_up = np.full(n, np.nan)
    bb_dn = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std = np.std(close[i-bb_period:i])
        bb_up[i] = sma[i] + bb_std * std
        bb_dn[i] = sma[i] - bb_std * std
        bb_width[i] = (bb_up[i] - bb_dn[i]) / sma[i] if sma[i] > 0 else 0
    
    # Bollinger Band width percentile for squeeze detection
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if not np.isnan(bb_width[i-lookback:i+1]).any():
            bb_width_percentile[i] = np.percentile(bb_width[i-lookback:i+1], 20)  # 20th percentile = squeeze
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(bb_period, vol_ma_period, 34) + lookback
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(bb_width_percentile[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volatility squeeze: BB width below 20th percentile of recent values
        volatility_squeeze = bb_width[i] < bb_width_percentile[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: uptrend, squeeze, volume, price breaks above R1
            if uptrend and volatility_squeeze and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend, squeeze, volume, price breaks below S1
            elif downtrend and volatility_squeeze and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below S1 or trend reverses or volatility expands
            if price < s1_aligned[i] or not uptrend or not volatility_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Exit short: price returns above R1 or trend reverses or volatility expands
            if price > r1_aligned[i] or not downtrend or not volatility_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSqueeze"
timeframe = "4h"
leverage = 1.0