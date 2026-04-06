#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Trend follows 12h EMA34.
# Long when Bull Power > 0 and increasing, Bear Power < 0, price > 12h EMA34, and volume > 1.5x average.
# Short when Bear Power < 0 and decreasing, Bull Power < 0, price < 12h EMA34, and volume > 1.5x average.
# This captures institutional buying/selling pressure with trend alignment, reducing false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.

name = "elder_ray_6h_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
BULL_EMA_PERIOD = 13   # For Bull/Bear Power calculation
TREND_EMA_PERIOD = 34  # 12h trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bull/Bear Power components
    ema_13 = calculate_ema(close, BULL_EMA_PERIOD)
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BULL_EMA_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Trend filter from 12h EMA
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Elder Ray signals with momentum
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bear_power_falling = bear_power[i] > bear_power[i-1] if i > 0 else False  # Bear power becomes more negative
        
        # Long conditions: Bull power positive AND rising, bear power negative, uptrend, volume
        long_signal = (bull_power[i] > 0) and bull_power_rising and (bear_power[i] < 0) and \
                      uptrend_12h and volume_ok
        
        # Short conditions: Bear power negative AND falling (more negative), bull power negative, downtrend, volume
        short_signal = (bear_power[i] > 0) and bear_power_falling and (bull_power[i] < 0) and \
                       downtrend_12h and volume_ok
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals