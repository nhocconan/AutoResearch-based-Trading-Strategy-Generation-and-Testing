#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly 52-week high/low breakout with volume confirmation on 1d timeframe
# Works in bull/bear because breakouts capture strong directional moves, volume filters weak signals,
# and 52-week levels represent significant structural support/resistance that works across regimes.
# Target: 30-80 trades over 4 years (7-20/year) to minimize fee drag.

name = "exp_12890_1d_weekly_52w_breakout_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
LOOKBACK_PERIOD = 52  # 52 weeks

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_52w_high_low(high, low, lookback):
    """Calculate 52-week high and low"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_52w = high_series.rolling(window=lookback, min_periods=lookback).max().values
    low_52w = low_series.rolling(window=lookback, min_periods=lookback).min().values
    return high_52w, low_52w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly 52-week high and low
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    
    high_52w, low_52w = calculate_52w_high_low(high_w, low_w, LOOKBACK_PERIOD)
    
    # Align to daily timeframe
    high_52w_aligned = align_htf_to_ltf(prices, df_weekly, high_52w)
    low_52w_aligned = align_htf_to_ltf(prices, df_weekly, low_52w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, LOOKBACK_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 52-week levels not available
        if np.isnan(high_52w_aligned[i]) or np.isnan(low_52w_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout above 52-week high or breakdown below 52-week low
        breakout_long = volume_ok and close[i] >= high_52w_aligned[i]
        breakout_short = volume_ok and close[i] <= low_52w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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

</think>