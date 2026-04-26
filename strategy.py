#!/usr/bin/env python3
"""
6h_WeeklyPivot_TrendReversal_v1
Hypothesis: Weekly pivot points (calculated from prior week's OHLC) act as strong support/resistance on 6h timeframe. 
In ranging/bear markets (2022-2025), price often reverses at weekly R1/S1 levels. 
Strategy: Long when price crosses above weekly S1 with bullish engulfing candle; 
Short when price crosses below weekly R1 with bearish engulfing candle. 
Volume confirmation (>1.5x 20-bar average) filters false breakouts. 
Only takes reversals, not breakouts, to avoid whipsaw in choppy markets. 
Discrete sizing (0.25) targets ~20-40 trades/year. Works in bull by catching reversals at support, 
in bear by catching reversals at resistance. Uses 1d HTF for weekly pivot calculation.
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
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivots from prior week's OHLC (using 1d data)
    # We need to group 1d data into weeks (Mon-Sun) and calculate pivots for prior week
    # For simplicity, we'll use prior week's high/low/close from 1d data
    # Align to 6h: each week's pivot applies to all 6h bars in the following week
    df_1d = df_1d.copy()
    df_1d['week'] = pd.to_datetime(df_1d.index).isocalendar().week
    df_1d['year'] = pd.to_datetime(df_1d.index).isocalendar().year
    
    # Group by year-week to get weekly OHLC
    weekly = df_1d.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate pivot points for each week (using prior week's OHLC)
    weekly['pivot'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['r1'] = 2 * weekly['pivot'] - weekly['low'].shift(1)
    weekly['s1'] = 2 * weekly['pivot'] - weekly['high'].shift(1)
    weekly['r2'] = weekly['pivot'] + (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s2'] = weekly['pivot'] - (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Forward fill weekly values to align with 1d data
    weekly_vals = weekly.set_index(['year', 'week'])
    
    # Map each 1d bar to its week/year and get pivot values
    df_1d['year'] = pd.to_datetime(df_1d.index).isocalendar().year
    df_1d['week'] = pd.to_datetime(df_1d.index).isocalendar().week
    
    # Merge weekly pivot data onto 1d dataframe
    df_1d = df_1d.merge(weekly[['year', 'week', 'pivot', 'r1', 's1', 'r2', 's2']], 
                        on=['year', 'week'], how='left')
    
    # Forward fill weekly values (each week's pivot applies to all days in that week)
    df_1d[['pivot', 'r1', 's1', 'r2', 's2']] = df_1d[['pivot', 'r1', 's1', 'r2', 's2']].ffill()
    
    # Extract arrays and align to 6h timeframe
    pivot_1d = df_1d['pivot'].values
    r1_1d = df_1d['r1'].values
    s1_1d = df_1d['s1'].values
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Bullish engulfing: current green candle engulfs prior red candle
    bullish_engulf = (close > open_) & (open_ < close_) & (close > close_) & (open_ < open_)
    # Bearish engulfing: current red candle engulfs prior green candle
    bearish_engulf = (close < open_) & (open_ > close_) & (close < close_) & (open_ > open_)
    # Fix: need open array
    open_ = prices['open'].values
    bullish_engulf = (close > open_) & (np.roll(close, 1) < np.roll(open_, 1)) & (close > np.roll(open_, 1)) & (open_ < np.roll(close, 1))
    bearish_engulf = (close < open_) & (np.roll(close, 1) > np.roll(open_, 1)) & (close < np.roll(open_, 1)) & (open_ > np.roll(close, 1))
    # Handle first bar
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of pivots (need 2 weeks), ATR (14), volume MA (20)
    start_idx = max(20, 14, 20)  # need at least 20 bars for volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        close_val = close[i]
        open_val = open_[i]
        high_val = high[i]
        low_val = low[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        bull_eng = bullish_engulf[i]
        bear_eng = bearish_engulf[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Entry conditions: reversal at weekly S1/R1 with engulfing candle + volume
        long_entry = (close_val > s1_val) and bull_eng and vol_conf
        short_entry = (close_val < r1_val) and bear_eng and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite pivot touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val > r1_val  # Stop or weekly R1 breakout (take profit)
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val < s1_val  # Stop or weekly S1 breakdown (take profit)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WeeklyPivot_TrendReversal_v1"
timeframe = "6h"
leverage = 1.0