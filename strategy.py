#!/usr/bin/env python3
"""
1d Bollinger Band Squeeze + Volume Spike + 1w Trend Filter
Hypothesis: Bollinger Band squeeze indicates low volatility and impending breakout.
Combined with volume spike (institutional interest) and 1-week trend filter (KAMA),
we capture explosive moves after consolidation periods. Works in both bull and bear
markets by trading breakouts in direction of higher timeframe trend.
Low trade frequency due to strict squeeze + volume + trend confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_length:
            er[i] = 0
        else:
            change_sum = np.sum(change[i-er_length+1:i+1])
            volatility_sum = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
            if volatility_sum > 0:
                er[i] = change_sum / volatility_sum
            else:
                er[i] = 0
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1w for trend filter
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, er_length=10, fast_ema=2, slow_ema=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    basis = np.zeros_like(close)
    dev = np.zeros_like(close)
    upper = np.zeros_like(close)
    lower = np.zeros_like(close)
    
    for i in range(n):
        if i < bb_length - 1:
            basis[i] = np.nan
            dev[i] = np.nan
        else:
            basis[i] = np.mean(close[i-bb_length+1:i+1])
            dev[i] = bb_mult * np.std(close[i-bb_length+1:i+1])
            upper[i] = basis[i] + dev[i]
            lower[i] = basis[i] - dev[i]
    
    # Bollinger Band Width (normalized)
    bb_width = np.where(basis != 0, (upper - lower) / basis, 0)
    
    # Bollinger Squeeze: BB width < 20-period average of BB width
    bb_width_ma = np.zeros_like(bb_width)
    for i in range(len(bb_width)):
        if i < 20:
            bb_width_ma[i] = np.mean(bb_width[max(0, i-19):i+1]) if i >= 0 else bb_width[i]
        else:
            bb_width_ma[i] = np.mean(bb_width[i-19:i+1])
    squeeze = bb_width < bb_width_ma
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        kama_val = kama_1w_aligned[i]
        sqz = squeeze[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: squeeze + volume spike + price above 1w KAMA (uptrend)
            if sqz and vol_ok and close[i] > kama_val:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + volume spike + price below 1w KAMA (downtrend)
            elif sqz and vol_ok and close[i] < kama_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below Bollinger middle band
            if close[i] < basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above Bollinger middle band
            if close[i] > basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BollingerSqueeze_VolumeSpike_1wKAMATrend"
timeframe = "1d"
leverage = 1.0