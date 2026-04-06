#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with daily EMA filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13). In bull markets: buy when Bull Power > 0 and rising.
# In bear markets: sell when Bear Power < 0 and falling. Daily EMA ensures higher timeframe trend alignment.
# Volume confirmation filters weak signals. Target: 50-150 total trades over 4 years.

name = "elder_ray_ema_volume_6h_v4"
timeframe = "6h"
leverage = 1.0

# Parameters
ER_EMA_PERIOD = 13      # EMA for Elder Ray calculation
DAILY_EMA_PERIOD = 50   # Daily EMA for trend filter
VOLUME_MA_PERIOD = 20   # Volume moving average
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.25      # Position size (25% of capital)
ATR_PERIOD = 14         # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5  # ATR multiplier for stop loss

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    daily_ema = calculate_ema(close_1d, DAILY_EMA_PERIOD)
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA(13) of close
    ema_er = calculate_ema(close, ER_EMA_PERIOD)
    bull_power = high - ema_er  # High - EMA
    bear_power = low - ema_er   # Low - EMA
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ER_EMA_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(daily_ema_aligned[i]) or np.isnan(ema_er[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
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
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > daily_ema_aligned[i]
        downtrend = close[i] < daily_ema_aligned[i]
        
        # Elder Ray signals with slope confirmation
        # Bull Power rising: current > previous
        bull_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        # Bear Power falling: current < previous
        bear_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        # Long signal: Bull Power > 0, rising, uptrend, volume confirmation
        long_signal = (bull_power[i] > 0) and bull_rising and uptrend and volume_ok
        # Short signal: Bear Power < 0, falling, downtrend, volume confirmation
        short_signal = (bear_power[i] < 0) and bear_falling and downtrend and volume_ok
        
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