#!/usr/bin/env python3
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
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(13) for trend filter
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on daily
    ema13 = ema13_1d
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align daily indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA13
        uptrend = close[i] > ema13_aligned[i]
        downtrend = close[i] < ema13_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Elder Ray signals: bull power > 0 and bear power < 0 for strength
        long_signal = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        short_signal = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        long_entry = long_signal and uptrend and vol_filter
        short_entry = short_signal and downtrend and vol_filter
        
        # Exit when Elder Ray weakens or trend reverses
        long_exit = (bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0) or not uptrend
        short_exit = (bull_power_aligned[i] >= 0 or bear_power_aligned[i] <= 0) or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0