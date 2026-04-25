#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dEMA50_TrendFilter_ATRStop
Hypothesis: Donchian(20) breakouts with volume spike confirmation and 1d EMA50 trend filter capture explosive moves while avoiding counter-trend whipsaw. ATR-based stoploss manages risk. Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR for stoploss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian(20) channels from 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    # Start index: need enough for Donchian (20) + volume MA (20) + EMA (50) + ATR (14)
    start_idx = max(lookback, 20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 1d EMA50 trend alignment
            long_breakout = curr_high > highest_high[i]
            short_breakout = curr_low < lowest_low[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_1d_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian low, trend turns bearish, or ATR stoploss hit
            atr_stop = entry_price - (2.0 * atr_1d_aligned[i])
            if curr_close < lowest_low[i] or curr_close < ema_50_1d_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian high, trend turns bullish, or ATR stoploss hit
            atr_stop = entry_price + (2.0 * atr_1d_aligned[i])
            if curr_close > highest_high[i] or curr_close > ema_50_1d_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dEMA50_TrendFilter_ATRStop"
timeframe = "4h"
leverage = 1.0