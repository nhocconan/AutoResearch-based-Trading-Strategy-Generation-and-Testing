#!/usr/bin/env python3
"""
4h Daily Williams %R Extreme + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions on daily timeframe.
In bull markets, buy oversold dips (%R < -80) in uptrend (price > EMA50).
In bear markets, sell overbought rallies (%R > -20) in downtrend (price < EMA50).
Volume spike confirms institutional participation. Works in both regimes via trend filter.
Target: 75-200 trades over 4 years (19-50/year). Discrete sizing: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    return wr

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R and EMA50 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams %R(14)
    wr_14_1d = calculate_williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    wr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_14_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for Williams %R, EMA, volume MA, ATR
    start_idx = max(14, 50, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Williams %R conditions
        wr_oversold = wr_14_1d_aligned[i] < -80  # Oversold
        wr_overbought = wr_14_1d_aligned[i] > -20  # Overbought
        
        if position == 0:
            # Look for entry signals - require: Williams %R extreme + volume spike + 1d EMA50 trend alignment
            long_entry = wr_oversold and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = wr_overbought and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: Williams %R returns to neutral, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            wr_exit = wr_14_1d_aligned[i] > -50  # Exit when %R returns to neutral
            if wr_exit or curr_close < ema_50_1d_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: Williams %R returns to neutral, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            wr_exit = wr_14_1d_aligned[i] < -50  # Exit when %R returns to neutral
            if wr_exit or curr_close > ema_50_1d_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyWilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0