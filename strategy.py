#!/usr/bin/env python3
"""
6h Elder Ray + Volume Spike + Regime Filter
Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure. Combined with volume spikes and a simple trend regime filter (price vs 50-period EMA), it captures strong momentum moves while avoiding chop. Works in both bull (buy power) and bear (bear power) markets by going long when bulls dominate and short when bears dominate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    ema = np.zeros_like(data)
    multiplier = 2 / (period + 1)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_elder_ray(high, low, close, ema_period=13):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    bull_power, bear_power, ema_13 = calculate_elder_ray(high_1d, low_1d, close_1d, ema_period=13)
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Calculate 50-period EMA for regime filter (trending market)
    ema_50 = calculate_ema(close_1d, 50)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema13_val = ema_13_aligned[i]
        ema50_val = ema_50_aligned[i]
        vol_ok = vol_spike[i]
        close_val = close[i]
        
        # Regime filter: trending market (price vs 50 EMA)
        # In bull regime: price > EMA50, in bear regime: price < EMA50
        is_bull_regime = close_val > ema50_val
        is_bear_regime = close_val < ema50_val
        
        if position == 0:
            # Enter long: bull power positive (buying pressure) + bull regime + volume spike
            if (bull_val > 0 and 
                is_bull_regime and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative (selling pressure) + bear regime + volume spike
            elif (bear_val < 0 and 
                  is_bear_regime and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative or regime changes
            if bull_val <= 0 or not is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive or regime changes
            if bear_val >= 0 or not is_bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeSpike_RegimeFilter"
timeframe = "6h"
leverage = 1.0