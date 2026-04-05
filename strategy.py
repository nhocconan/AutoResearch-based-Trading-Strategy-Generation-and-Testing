#!/usr/bin/env python3
"""
Experiment #10399: 6h Williams Alligator + Elder Ray + Volume Confirmation
Hypothesis: Williams Alligator identifies trend direction and alignment, Elder Ray confirms momentum strength,
and volume filter ensures institutional participation. Works in trending markets (both bull/bear) by
filtering for strong directional moves with volume confirmation. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10399_6h_williams_alligator_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ELDER_RAY_PERIOD = 13
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smma(values, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    sma = np.full_like(values, np.nan, dtype=float)
    if len(values) >= period:
        sma[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            sma[i] = (sma[i-1] * (period-1) + values[i]) / period
    return sma

def calculate_williams_alligator(high, low, close, period):
    """Calculate Williams Alligator (Jaw, Teeth, Lips)"""
    median_price = (high + low) / 2
    jaw = calculate_smma(median_price, period * 3)  # Slowest
    teeth = calculate_smma(median_price, period * 2)  # Medium
    lips = calculate_smma(median_price, period)     # Fastest
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray (Bull Power, Bear Power)"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for higher timeframe trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_williams_alligator(high, low, close, ALLIGATOR_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD * 3, ELDER_RAY_PERIOD, 20, 50) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips[i] > teeth[i] > jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        bearish_alignment = (lips[i] < teeth[i] < jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > np.nanpercentile(bull_power[max(0, i-50):i+1], 60) if i >= 50 else bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0 and abs(bear_power[i]) > np.nanpercentile(abs(bear_power[max(0, i-50):i+1]), 60) if i >= 50 else bear_power[i] < 0
        
        # Higher timeframe trend filter
        above_12h_ema = close[i] > ema_12h_aligned[i]
        below_12h_ema = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = bullish_alignment and strong_bull_power and volume_spike and above_12h_ema
        short_entry = bearish_alignment and strong_bear_power and volume_spike and below_12h_ema
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals