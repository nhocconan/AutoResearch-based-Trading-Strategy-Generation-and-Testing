#!/usr/bin/env python3
"""
Experiment #9679: 6h Elder Ray + ADX Regime Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) identifies bull/bear pressure, while ADX filters regime.
In trending markets (ADX > 25): trade with Elder Ray direction. In ranging markets (ADX < 20): fade extremes.
Combines momentum and mean reversion to work in both bull and bear markets.
Targets 100-180 total trades over 4 years (25-45/year) for statistical validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9679_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for Elder Ray (13-period EMA)
    ema = calculate_ema(close, ELDER_RAY_PERIOD)
    
    # Bull Power = High - EMA
    bull_power = high - ema
    
    # Bear Power = Low - EMA
    bear_power = low - ema
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
        
        # Regime filters
        trending = adx[i] >= ADX_TREND_THRESHOLD   # Trending market
        ranging = adx[i] <= ADX_RANGE_THRESHOLD    # Ranging market
        
        # Trend following signals (ADX > 25): follow Elder Ray direction
        trend_long = trending and bull_power[i] > 0
        trend_short = trending and bear_power[i] < 0
        
        # Mean reversion signals (ADX < 20): fade extreme Elder Ray readings
        # In ranging markets, fade when Bull/Bear Power reaches extremes
        range_long = ranging and bear_power[i] < 0 and bear_power[i] < np.percentile(bear_power[max(0, i-50):i+1], 10)
        range_short = ranging and bull_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 90)
        
        # Entry conditions
        long_entry = trend_long or range_long
        short_entry = trend_short or range_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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