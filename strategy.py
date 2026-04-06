#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 1-day EMA trend filter and volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Long when Bull Power > 0 and Bear Power rising (less negative) in uptrend.
# Short when Bear Power < 0 and Bull Power falling (less positive) in downtrend.
# Uses 1-day EMA for trend direction, 6-hour for entries, volume for confirmation.
# Designed for ~100 total trades over 4 years (25/year) to avoid fee drain.
# Works in bull (strong bull power) and bear (strong bear power) markets.

name = "exp_13707_6h_elderray1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Elder Ray components: Bull Power and Bear Power
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Elder Ray signals with smoothing (check current vs previous)
        if i > 0:
            bull_rising = bull_power[i] > bull_power[i-1]  # Bull power increasing
            bull_falling = bull_power[i] < bull_power[i-1]  # Bull power decreasing
            bear_rising = bear_power[i] > bear_power[i-1]  # Bear power increasing (less negative)
            bear_falling = bear_power[i] < bear_power[i-1]  # Bear power decreasing (more negative)
        else:
            bull_rising = bull_falling = bear_rising = bear_falling = False
        
        # Long signal: Bull Power > 0 AND Bear Power rising (less negative) in uptrend with volume
        long_signal = volume_ok and above_ema and (bull_power[i] > 0) and bear_rising
        
        # Short signal: Bear Power < 0 AND Bull Power falling (less positive) in downtrend with volume
        short_signal = volume_ok and below_ema and (bear_power[i] < 0) and bull_falling
        
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
            # Exit long when Bear Power turns negative (bearish momentum)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when Bull Power turns negative (bullish momentum)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals