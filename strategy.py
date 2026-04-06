#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray Index (Bull/Bear Power) with EMA13 filter.
# Goes long when Bull Power > 0 and price above EMA13, short when Bear Power < 0 and price below EMA13.
# Uses 12h EMA13 as trend filter and Elder Ray to measure bull/bear strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (strong bull power) and bear (strong bear power) markets by following institutional pressure.

name = "exp_13779_6d_elder_ray_ema13_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA_PERIOD = 13
TREND_EMA_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_elder_ray(high, low, close, ema_period):
    """Calculate Elder Ray Index: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Elder Ray and EMA trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Elder Ray Index
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    bull_power, bear_power = calculate_elder_ray(high_12h, low_12h, close_12h, ELDER_RAY_EMA_PERIOD)
    
    # Calculate 12h EMA for trend filter
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h data for ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA_PERIOD, TREND_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Elder Ray signals with EMA filter
        bull_strong = bull_power_aligned[i] > 0
        bear_strong = bear_power_aligned[i] < 0
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        long_signal = bull_strong and above_ema
        short_signal = bear_strong and below_ema
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when bear power becomes positive or price below EMA
            if bear_power_aligned[i] >= 0 or close[i] <= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when bull power becomes negative or price above EMA
            if bull_power_aligned[i] <= 0 or close[i] >= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals