#!/usr/bin/env python3
"""
4h_Vortex_VortexTrend_1dFilter
Strategy: 4h Vortex Indicator with 1d trend filter and volume confirmation.
Long: VI+ > VI- + VI+ rising + 1d uptrend + volume > 1.5x 10-period average
Short: VI- > VI+ + VI- rising + 1d downtrend + volume > 1.5x 10-period average
Exit: Vortex cross reversal or trend reversal
Position size: 0.25
Designed to catch trend momentum while filtering chop.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Vortex Indicator (VI)
    # VI+ = EMA(|High - Prev Low|, n) / EMA(True Range, n)
    # VI- = EMA(|Low - Prev High|, n) / EMA(True Range, n)
    vm = 10  # Vortex period
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(low, 1))
    tr3 = np.abs(low - np.roll(high, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Vortex movement
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Smooth with EMA
    tr_ema = pd.Series(tr).ewm(span=vm, adjust=False).mean().values
    vm_plus_ema = pd.Series(vm_plus).ewm(span=vm, adjust=False).mean().values
    vm_minus_ema = pd.Series(vm_minus).ewm(span=vm, adjust=False).mean().values
    
    # Avoid division by zero
    vi_plus = np.where(tr_ema != 0, vm_plus_ema / tr_ema, 0)
    vi_minus = np.where(tr_ema != 0, vm_minus_ema / tr_ema, 0)
    
    # Calculate 1d trend (close > open = uptrend, close < open = downtrend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values  # 1 for up, 0 for down
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 4h volume average (10-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    volume_ma10_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma10_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(vm*2, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(volume_ma10_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma10_4h_aligned[i])
        
        # Trend filter: 1d bullish/bearish
        trend_up = trend_1d_aligned[i] > 0.5  # 1d close > open
        trend_down = trend_1d_aligned[i] < 0.5  # 1d close < open
        
        # Vortex signals
        vi_cross_up = vi_plus[i-1] <= vi_minus[i-1] and vi_plus[i] > vi_minus[i]  # VI+ crosses above VI-
        vi_cross_down = vi_minus[i-1] <= vi_plus[i-1] and vi_minus[i] > vi_plus[i]  # VI- crosses above VI+
        vi_plus_rising = vi_plus[i] > vi_plus[i-1]  # VI+ rising
        vi_minus_rising = vi_minus[i] > vi_minus[i-1]  # VI- rising
        
        # Entry signals
        if position == 0:
            # Long: VI+ > VI- + VI+ rising + volume filter + 1d uptrend
            if vi_plus[i] > vi_minus[i] and vi_plus_rising and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ + VI- rising + volume filter + 1d downtrend
            elif vi_minus[i] > vi_plus[i] and vi_minus_rising and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ or 1d trend turns down
            if vi_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- or 1d trend turns up
            if vi_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Vortex_VortexTrend_1dFilter"
timeframe = "4h"
leverage = 1.0