#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour breakout strategy using 4h Donchian channels and 1d EMA trend filter with volume confirmation.
# Uses higher timeframes for direction (4h/1d) and 1h for precise entry timing to reduce false breakouts.
# Includes session filter (08-20 UTC) to avoid low-liquidity periods. Targets 60-150 trades over 4 years.
# Works in bull markets (breakouts above 4h high with uptrend) and bear markets (breakdowns below 4h low with downtrend).

name = "exp_13294_1h_donchian4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20    # 4h Donchian period
EMA_PERIOD = 50         # 1d EMA for trend filter
VOLUME_MA_PERIOD = 20   # Volume moving average
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.20      # Position size (20% of capital)
ATR_PERIOD = 14         # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0  # Stop loss at 2x ATR

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
    
    # Calculate 4h Donchian channels (highest high, lowest low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-compute session hours (08-20 UTC) to avoid low-liquidity periods
    # prices.index is already DatetimeIndex, so .hour works directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all needed periods)
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1:  # Long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation: current volume > threshold * average volume
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price relative to 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals: price breaks 4h Donchian levels with volume and trend alignment
        breakout_up = volume_ok and uptrend and (high[i] > donchian_high_aligned[i])
        breakout_down = volume_ok and downtrend and (low[i] < donchian_low_aligned[i])
        
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