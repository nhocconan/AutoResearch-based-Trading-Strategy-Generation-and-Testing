#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly EMA trend filter and volume confirmation.
# The 1-day timeframe reduces noise and transaction costs while capturing major trends.
# Weekly EMA ensures alignment with higher timeframe momentum to avoid counter-trend trades.
# Volume confirmation filters out false breakouts. Target: 30-100 total trades over 4 years.

name = "exp_13344_1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20      # Daily Donchian period
EMA_PERIOD = 20           # Weekly EMA period
VOLUME_MA_PERIOD = 20     # Volume moving average
VOLUME_THRESHOLD = 1.5    # Volume must be 1.5x average
SIGNAL_SIZE = 0.25        # Position size (25% of capital)
ATR_PERIOD = 14           # ATR period for stop loss
ATR_STOP_MULTIPLIER = 2.0 # Stop loss multiplier

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Load daily data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band = highest high over period, Lower band = lowest low over period
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper = high_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lower = low_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Align Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals using Donchian channels
        breakout_up = volume_ok and uptrend and (high[i] > upper_aligned[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < lower_aligned[i-1])
        
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