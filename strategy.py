#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Supertrend (ATR=10, multiplier=3) with 1-day volume confirmation and
# 1-week trend filter. Supertrend identifies trend direction with dynamic support/resistance.
# Volume confirms institutional participation. Weekly EMA ensures alignment with higher timeframe
# momentum to avoid counter-trend trades. Target: 50-150 total trades over 4 years.
# Works in bull markets (uptrend + buy signal) and bear markets (downtrend + sell signal).

name = "exp_13327_6h_supertrend_1d_vol_1w_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
EMA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 10

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(upper[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data ONCE before loop for volume
    df_1d = get_htf_data(prices, '1d')
    # Load 1-week data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-day volume MA
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend
    supertrend, direction = calculate_supertrend(high, low, close, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, VOLUME_MA_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(supertrend[i]) or np.isnan(volume_ma_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
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
        
        # Volume confirmation: current 6h volume > 1.5x 1-day average volume
        # Scale 1-day volume to 6h: 1 day = 4 x 6h bars
        volume_ma_6h_scaled = volume_ma_1d_aligned[i] / 4.0
        volume_ok = volume[i] > (volume_ma_6h_scaled * VOLUME_THRESHOLD) if not np.isnan(volume_ma_6h_scaled) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Supertrend signal
        st_buy = direction[i] == 1 and close[i] > supertrend[i]
        st_sell = direction[i] == -1 and close[i] < supertrend[i]
        
        # Generate signals
        if position == 0:
            if st_buy and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif st_sell and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals