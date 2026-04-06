#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour range breakout with 4-hour trend filter and daily volume confirmation.
# In bull markets, buy breakouts above 4-hour resistance; in bear markets, sell breakouts below 4-hour support.
# Daily volume filter ensures only significant breakouts trigger entries. 4-hour trend avoids counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours. Target: 100-150 total trades over 4 years.

name = "exp_13274_1h_range_breakout_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RANGE_PERIOD = 20          # Lookback for 1h range
TREND_PERIOD = 50          # 4h EMA for trend filter
VOLUME_MA_PERIOD = 20      # Daily volume moving average
VOLUME_THRESHOLD = 1.5     # Volume spike multiplier
SIGNAL_SIZE = 0.20         # Position size (20% of capital)
ATR_PERIOD = 14            # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0  # Stop loss distance

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h range (highest high / lowest low over period)
    highest_high = pd.Series(high).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RANGE_PERIOD, TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h EMA or 1d volume MA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
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
        
        # Volume confirmation: current 1h volume > 1.5x daily average volume
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Trend filter: price above/below 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Breakout signals: price breaks 1h range with volume and trend confirmation
        breakout_up = volume_ok and uptrend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < lowest_low[i-1])
        
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