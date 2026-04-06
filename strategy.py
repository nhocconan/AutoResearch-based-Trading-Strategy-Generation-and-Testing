#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Uses Donchian channel for breakout detection, 12h EMA for trend direction, and volume spike for confirmation.
# Only trades in trend direction with volume > 1.5x 20-period average.
# Includes ATR-based stop loss and time-based exit (max 10 bars).
# Designed for 4h timeframe to target 20-50 trades/year with low turnover.
# Should work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets.

name = "exp_13623_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 10

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_held = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Update bars held
        if position != 0:
            bars_held += 1
        
        # Check stoploss or max hold time
        if position == 1:  # long position
            if close[i] <= stop_price or bars_held >= MAX_HOLD_BARS:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or bars_held >= MAX_HOLD_BARS:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA
        uptrend = ema_12h_aligned[i] > 0
        downtrend = ema_12h_aligned[i] < 0
        
        # Donchian breakout signals
        # Avoid lookback by checking current and previous values
        if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            # Long signal: price breaks above Donchian upper in uptrend with volume
            long_signal = volume_ok and uptrend and close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
            
            # Short signal: price breaks below Donchian lower in downtrend with volume
            short_signal = volume_ok and downtrend and close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        else:
            long_signal = False
            short_signal = False
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay in long position
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Stay in short position
            signals[i] = -SIGNAL_SIZE
    
    return signals