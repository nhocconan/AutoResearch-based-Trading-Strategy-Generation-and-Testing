#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13954_1h_ema13_ema48_vol"
timeframe = "1h"
leverage = 1.0

# Hypothesis: 1h EMA crossover (13/48) with volume confirmation and session filter (08-20 UTC).
# Uses EMA crossover for momentum: EMA13 > EMA48 = bullish, EMA13 < EMA48 = bearish.
# Entry on crossover with volume > 1.5x average during active session.
# Designed for low trade frequency to avoid fee drag: ~20-30 trades/year.
# Works in bull (bullish crossover) and bear (bearish crossover) markets.

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA indicators
    ema_13 = calculate_ema(close, 13)
    ema_48 = calculate_ema(close, 48)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(48, 20) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(ema_13[i]) or np.isnan(ema_48[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema_bullish = ema_13[i] > ema_48[i]
        ema_bearish = ema_13[i] < ema_48[i]
        
        # Previous EMA values for crossover detection
        ema_13_prev = ema_13[i-1]
        ema_48_prev = ema_48[i-1]
        ema_bullish_prev = ema_13_prev > ema_48_prev
        ema_bearish_prev = ema_13_prev < ema_48_prev
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Crossover detection
        bullish_cross = ema_bullish and not ema_bullish_prev  # EMA13 crosses above EMA48
        bearish_cross = ema_bearish and not ema_bearish_prev  # EMA13 crosses below EMA48
        
        # Generate signals
        if position == 0:
            if bullish_cross and volume_ok:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif bearish_cross and volume_ok:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish crossover
            if bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on bullish crossover
            if bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals