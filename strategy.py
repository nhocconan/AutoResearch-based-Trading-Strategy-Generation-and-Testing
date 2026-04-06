#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 12-hour directional filter and volume confirmation.
# Uses 12h Donchian to establish trend direction (bullish if price > 12h DC upper, bearish if < lower).
# Entry occurs when 6h price breaks the 6h Donchian channel in the direction of the 12h trend,
# confirmed by volume > 1.5x 20-period MA. This captures momentum continuations in both bull and bear markets
# while avoiding counter-trend whipsaws. Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13219_6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_DONCHIAN_PERIOD = 20  # for 12h trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    highest_high_12h = pd.Series(high_12h).rolling(window=TREND_DONCHIAN_PERIOD, min_periods=TREND_DONCHIAN_PERIOD).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=TREND_DONCHIAN_PERIOD, min_periods=TREND_DONCHIAN_PERIOD).min().values
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h trend data not available
        if np.isnan(highest_high_12h_aligned[i]) or np.isnan(lowest_low_12h_aligned[i]):
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
        
        # Trend filter: 12h Donchian breakout direction
        # Bullish trend: price above 12h Donchian upper
        # Bearish trend: price below 12h Donchian lower
        bullish_trend = close[i] > highest_high_12h_aligned[i]
        bearish_trend = close[i] < lowest_low_12h_aligned[i]
        
        # Breakout signals: 6h price breaks 6h Donchian in trend direction
        breakout_up = volume_ok and bullish_trend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and bearish_trend and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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