#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d Elder Ray (bull/bear power) with 12h EMA trend filter and volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Go long when Bull Power > 0 and increasing, price above 12h EMA20, and volume above average.
# Go short when Bear Power < 0 and decreasing, price below 12h EMA20, and volume above average.
# Uses ATR-based stop loss to manage risk.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drift.
# Elder Ray captures institutional buying/selling pressure, EMA20 filters short-term trend, volume confirms strength.

name = "exp_13827_6h_elderray12h_ema20_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13  # EMA period for Elder Ray
EMA_TREND_PERIOD = 20  # EMA for trend filter
VOLUME_MA_PERIOD = 20  # Volume moving average
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.25  # Position size (25% of capital)
ATR_PERIOD = 14  # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0  # ATR multiplier for stop loss

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1d data for Elder Ray calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, ELDER_RAY_PERIOD)
    
    # Calculate Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_1d
    bear_power = low_1d - ema_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h data for EMA trend filter, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for trend filter on 6h data
    ema_trend = calculate_ema(close, EMA_TREND_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_trend[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 6h EMA
        above_ema = close[i] > ema_trend[i]
        below_ema = close[i] < ema_trend[i]
        
        # Elder Ray signals with momentum (current > previous)
        bull_power_current = bull_power_aligned[i]
        bear_power_current = bear_power_aligned[i]
        bull_power_prev = bull_power_aligned[i-1] if i > 0 else 0
        bear_power_prev = bear_power_aligned[i-1] if i > 0 else 0
        
        bull_power_increasing = bull_power_current > bull_power_prev
        bear_power_decreasing = bear_power_current < bear_power_prev
        
        # Long signal: Bull Power > 0 and increasing, price above EMA, volume confirmation
        long_signal = (bull_power_current > 0) and bull_power_increasing and above_ema and volume_ok
        
        # Short signal: Bear Power < 0 and decreasing, price below EMA, volume confirmation
        short_signal = (bear_power_current < 0) and bear_power_decreasing and below_ema and volume_ok
        
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
            # Exit long on Bear Power turning negative (bearish pressure)
            if bear_power_current < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Bull Power turning positive (bullish pressure)
            if bull_power_current > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals