#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour volume-weighted breakout with 4-hour trend filter and daily volume confirmation.
# Uses 4-hour trend direction to filter breakouts, daily volume to confirm institutional interest.
# Entry only during active London/New York session (08-20 UTC) to avoid low-volume noise.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost.
# Works in bull markets by capturing uptrend continuations and in bear markets by catching breakdowns.

name = "exp_13174_1h_vol_weighted_breakout_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BREAKOUT_PERIOD = 15          # Lookback for high/low breakout
VOLUME_MA_PERIOD = 20         # Volume moving average
VOLUME_THRESHOLD = 1.8        # Volume must be 1.8x average
TREND_PERIOD = 20             # 4h EMA for trend filter
SIGNAL_SIZE = 0.20            # Position size (20% of capital)
ATR_PERIOD = 14               # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5     # Stop loss distance

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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA for institutional confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Breakout channels (15-period high/low)
    highest_high = pd.Series(high).rolling(window=BREAKOUT_PERIOD, min_periods=BREAKOUT_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=BREAKOUT_PERIOD, min_periods=BREAKOUT_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (London + NY overlap)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BREAKOUT_PERIOD, VOLUME_MA_PERIOD, TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if indicators not ready
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
        
        # Volume confirmation (1h and 1d)
        volume_ok_1h = not np.isnan(volume_ma[i]) and volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        volume_ok_1d = not np.isnan(volume_ma_1d_aligned[i]) and volume_1d[-1] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if len(volume_1d) > 0 else False
        volume_ok = volume_ok_1h and volume_ok_1d
        
        # Trend filter: 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Breakout signals (using prior bar's high/low to avoid look-ahead)
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