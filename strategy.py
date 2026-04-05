#!/usr/bin/env python3
"""
Experiment #7835: 6-hour Donchian breakout with weekly pivot direction and volume confirmation.
Hypothesis: Price breaking beyond 20-period high/low on 6h with volume >1.8x 20-period MA and aligned weekly trend (via daily pivot direction) captures sustained moves while avoiding whipsaw. The weekly pivot (derived from 1d data) provides directional bias from higher timeframe to reduce false breakouts in both bull and bear markets. Targets 50-150 trades over 4 years with controlled risk via ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7835_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
PIVOT_LOOKBACK = 5  # days for pivot calculation
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_pivot_direction(daily_data):
    """Calculate pivot direction from daily OHLC data.
    Returns: 1 for bullish bias (price above pivot), -1 for bearish bias (price below pivot)
    """
    if len(daily_data) < PIVOT_LOOKBACK:
        return np.zeros(len(daily_data))
    
    high = daily_data['high'].values
    low = daily_data['low'].values
    close = daily_data['close'].values
    
    # Calculate daily pivots using standard formula
    pivot = (high + low + close) / 3.0
    
    # Determine bias: price above pivot = bullish, below = bearish
    bias = np.where(close > pivot, 1, -1)
    
    # Smooth the bias to avoid whipsaw - require 3-day confirmation
    bias_series = pd.Series(bias)
    bias_confirmed = bias_series.rolling(window=3, min_periods=3).apply(
        lambda x: 1 if np.all(x == 1) else (-1 if np.all(x == -1) else 0), raw=False
    ).fillna(0).values
    
    return bias_confirmed

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d data to calculate weekly pivot direction
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot direction from 1d data (proxy for weekly bias)
    pivot_bias_1d = calculate_pivot_direction({
        'high': df_1d['high'],
        'low': df_1d['low'],
        'close': df_1d['close']
    })
    pivot_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price channel (Donchian)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_bias_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d pivot (proxy for weekly)
        bull_bias = pivot_bias_1d_aligned[i] == 1   # price above pivot = bullish
        bear_bias = pivot_bias_1d_aligned[i] == -1  # price below pivot = bearish
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond channel bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals