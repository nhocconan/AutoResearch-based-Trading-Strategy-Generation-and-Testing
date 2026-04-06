#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 34-period EMA trend filter with 4-hour Donchian channel (20) breakout and volume confirmation.
# Uses 12-hour EMA for higher timeframe trend confirmation. Trades only in direction of higher timeframe trend
# to avoid counter-trend whipsaws. Volume requirement ensures breakouts have institutional participation.
# Stoploss at 2x ATR manages risk. Designed for 40-80 trades per year to minimize fee drag.

name = "exp_13313_4h_ema34_donchian20_vol_12hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
EMA_FAST = 34      # Fast EMA for trend
EMA_SLOW = 12      # Slow EMA for higher timeframe trend (12h EMA on 4h chart)
DONCHIAN_PERIOD = 20  # Donchian channel period
VOLUME_MA = 20     # Volume moving average
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.25   # Position size (25% of capital)
ATR_PERIOD = 14    # ATR period for stoploss
ATR_STOP_MULT = 2.0  # ATR multiplier for stoploss

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Load 12h data ONCE before loop for higher timeframe trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_SLOW)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for trend filter (fast)
    ema_fast = calculate_ema(close, EMA_FAST)
    
    # Donchian channels
    donch_up, donch_low = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, DONCHIAN_PERIOD, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(donch_up[i]) or np.isnan(donch_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Determine trend from 12h EMA and 4h EMA
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        uptrend_4h = close[i] > ema_fast[i]
        downtrend_4h = close[i] < ema_fast[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals
        breakout_up = volume_ok and uptrend_12h and uptrend_4h and (high[i] > donch_up[i-1])
        breakout_down = volume_ok and downtrend_12h and downtrend_4h and (low[i] < donch_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULT * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULT * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals