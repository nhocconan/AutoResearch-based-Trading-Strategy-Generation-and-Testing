#!/usr/bin/env python3
"""
12h_KAMA_Direction_1dATR_Regime_VolumeConfirm
Hypothesis: On 12h timeframe, KAMA (Kaufman Adaptive Moving Average) direction combined with 1d ATR-based regime filter and volume confirmation.
KAMA adapts to market noise, providing smooth trend direction in both bull and bear markets. 1d ATR regime identifies trending vs ranging markets - 
we trade only in trending regimes (ATR ratio > threshold) to avoid whipsaws. Volume confirmation ensures institutional participation.
Designed for 12-37 trades/year with discrete position sizing (0.25) to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average with min_periods"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    close_series = pd.Series(close)
    direction = np.abs(close_series.diff(period))
    volatility = close_series.diff().abs().rolling(window=period, min_periods=period).sum()
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First TR is just high-low
    
    atr = np.zeros_like(high)
    atr[0] = true_range[0]
    for i in range(1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ATR regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 12h KAMA direction filter
    kama_12h = calculate_kama(close, period=10, fast=2, slow=30)
    
    # 1d ATR regime: current ATR vs 20-period average (trending when ATR expanding)
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d  # >1.0 indicates expanding volatility (trending)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA (10) + ATR (20) + volume MA (20)
    start_idx = max(30, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_12h[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama_12h[i]
        atr_ratio = atr_ratio_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: KAMA direction + trending regime (ATR ratio > 1.1) + volume spike
            long_condition = (curr_close > curr_kama) and (atr_ratio > 1.1) and volume_spike[i]
            short_condition = (curr_close < curr_kama) and (atr_ratio > 1.1) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below KAMA or regime turns ranging
            if curr_close < curr_kama or atr_ratio < 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above KAMA or regime turns ranging
            if curr_close > curr_kama or atr_ratio < 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_1dATR_Regime_VolumeConfirm"
timeframe = "12h"
leverage = 1.0