#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (bull/bear power) with daily EMA trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA.
# Combines trend following (EMA direction) with momentum (power divergence). Works in bull markets via
# sustained bull power > 0 and in bear markets via sustained bear power < 0. Volume confirms institutional
# participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "elder_ray_ema_volume_6h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_PERIOD = 20          # Daily EMA for trend and Elder Ray
VOLUME_MA_PERIOD = 20    # Volume moving average
VOLUME_THRESHOLD = 1.5   # Volume must be 1.5x average
SIGNAL_SIZE = 0.25       # Position size (25% of capital)
ATR_PERIOD = 14          # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0 # Stop loss at 2x ATR

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EWMA)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # First TR is just high-low (no previous close)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter and Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
        bull_power = high[i] - ema_1d_aligned[i]
        bear_power = low[i] - ema_1d_aligned[i]
        
        # Entry conditions
        # Long: bull power > 0 AND volume confirmation
        # Short: bear power < 0 AND volume confirmation
        long_signal = bull_power > 0 and volume_ok
        short_signal = bear_power < 0 and volume_ok
        
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