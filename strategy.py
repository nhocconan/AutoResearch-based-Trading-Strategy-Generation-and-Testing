#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4h Donchian channel breakout with volume confirmation
# and 1-day trend filter (price vs 50-day EMA). Uses 4h for signal direction and 1h for
# entry timing precision. Includes session filter (08-20 UTC) to reduce noise trades.
# Targets 80-150 total trades over 4 years (20-38/year) for statistical validity.
# Works in bull markets (breakouts above 4h high) and bear markets (breakdowns below 4h low).
# Position size fixed at 0.20 to manage drawdown and reduce turnover.

name = "exp_13334_1h_donchian4h_vol_ema1d_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20  # 4h Donchian period
EMA_PERIOD = 50       # 1-day EMA for trend filter
VOLUME_MA_PERIOD = 20 # Volume moving average
VOLUME_THRESHOLD = 1.5 # Volume spike threshold
SIGNAL_SIZE = 0.20    # Fixed position size (20% of capital)
ATR_PERIOD = 14       # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0 # Stop loss multiplier

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
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Align Donchian levels to 1h timeframe (with shift(1) for completed bars only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all lookbacks)
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if indicators not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
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
        
        # Volume confirmation: current volume > threshold * average volume
        volume_ok = not np.isnan(volume_ma[i]) and volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below 1-day EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals using 4h Donchian channels
        # Note: using i-1 to avoid look-ahead (current bar high/low vs previous Donchian levels)
        breakout_up = volume_ok and uptrend and (high[i] > donchian_high_aligned[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < donchian_low_aligned[i-1])
        
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