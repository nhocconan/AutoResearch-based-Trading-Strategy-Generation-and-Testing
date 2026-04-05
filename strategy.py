#!/usr/bin/env python3
"""
exp_7279_6h_donchian20_12h_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h weekly pivot (from 1d HTF) direction filter and volume confirmation.
In trending markets (price > weekly pivot R1): continuation breakouts above R2.
In ranging markets (price between R1/S1): mean reversion at Donchian extremes with volume confirmation.
Uses weekly pivot levels calculated from 1d HTF data for regime and 6h volume for confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to pivot-defined trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7279_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~32 hours

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for weekly pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate weekly pivot points from 12h data (using prior week's high/low/close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate prior week's OHLC (5 * 12h bars = 60h ≈ 2.5 days, use prior 5 bars for weekly)
    # We'll use prior 5 periods to approximate weekly
    lookback = 5
    if len(high_12h) >= lookback:
        # Rolling window for weekly high/low/close
        weekly_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().shift(1).values
        weekly_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().shift(1).values
        weekly_close = pd.Series(close_12h).rolling(window=lookback, min_periods=lookback).last().shift(1).values
        
        # Calculate pivot points
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        
        # Align to LTF (6h)
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
        r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
        s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
        r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
        s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    else:
        # Not enough data
        pivot_aligned = r1_aligned = s1_aligned = r2_aligned = s2_aligned = r3_aligned = s3_aligned = np.full(n, np.nan)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on pivot levels
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        between_r1_s1 = (close[i] >= s1_aligned[i]) and (close[i] <= r1_aligned[i])
        
        # Fade at Donchian extremes in ranging market (between R1/S1)
        fade_long = between_r1_s1 and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = between_r1_s1 and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market
        continuation_long = above_r1 and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_s1 and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals