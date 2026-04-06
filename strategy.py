#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13/8/5 SMAs) with Elder Ray power signals filtered by 1d EMA trend.
# Alligator identifies trend state (sleeping/awake/eating). Elder Ray (bull/bear power) measures trend strength.
# 1d EMA ensures trading with higher timeframe momentum. Works in bull/bear by only taking signals in trend direction.
# Target: 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.

name = "exp_13059_6h_alligator_elder_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13  # Blue line
ALLIGATOR_PERIOD_TEETH = 8  # Red line
ALLIGATOR_PERIOD_LIPS = 5   # Green line
EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_sma(arr, period):
    """Calculate Simple Moving Average"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three SMAs
    jaw = calculate_sma(close, ALLIGATOR_PERIOD_JAW)  # Blue (13)
    teeth = calculate_sma(close, ALLIGATOR_PERIOD_TEETH)  # Red (8)
    lips = calculate_sma(close, ALLIGATOR_PERIOD_LIPS)  # Green (5)
    
    # Elder Ray Power
    bull_power = high - teeth  # High minus Teeth (red line)
    bear_power = low - teeth   # Low minus Teeth (red line)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Alligator conditions: check if lines are not intertwined (trending)
        # Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_up = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        alligator_down = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        
        # Elder Ray signals with Alligator filter
        # Long: bull power positive AND alligator aligned up
        # Short: bear power negative AND alligator aligned down
        long_signal = bull_power[i] > 0 and alligator_up
        short_signal = bear_power[i] < 0 and alligator_down
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal and downtrend:
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