#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Strategy: 6h High-Low Range Breakout with 1d Direction Filter
    Timeframe: 6h
    Hypothesis: In trending markets (identified by 1d EMA cross), 6h range breakouts 
    capture momentum with high win rate. Range breakouts filter out chop, and 
    EMA filter ensures we only trade in direction of higher timeframe trend.
    Works in both bull (buy breakouts above range in uptrend) and bear 
    (sell breakdowns below range in downtrend) markets.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context (1d) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA crossover for trend direction (EMA21 vs EMA50)
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h range (high-low of previous bar) for breakout detection
    # We use previous bar's range to avoid look-ahead
    range_high = np.maximum(high[:-1], np.roll(high, 1)[:-1])  # previous bar high
    range_low = np.minimum(low[:-1], np.roll(low, 1)[:-1])    # previous bar low
    # Pad first element
    range_high = np.concatenate([[range_high[0]], range_high])
    range_low = np.concatenate([[range_low[0]], range_low])
    
    # Breakout levels: previous bar's high/low
    breakout_level_high = range_high
    breakout_level_low = range_low
    
    # Volume filter: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC) - avoid low liquidity periods
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA21 > EMA50 = uptrend, EMA21 < EMA50 = downtrend
        uptrend = ema21_1d_aligned[i] > ema50_1d_aligned[i]
        downtrend = ema21_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Volume filter: current volume significantly above average
        volume_filter = vol_ma[i] > 0 and volume[i] > vol_ma[i] * 1.5
        
        # Breakout signals: price breaks previous bar's range
        breakout_up = close[i] > breakout_level_high[i]
        breakout_down = close[i] < breakout_level_low[i]
        
        # Long conditions: uptrend + volume + upward breakout
        long_condition = (uptrend and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: downtrend + volume + downward breakout
        short_condition = (downtrend and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal (EMA cross)
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RangeBreakout_EMA21_50_VolumeFilter"
timeframe = "6h"
leverage = 1.0