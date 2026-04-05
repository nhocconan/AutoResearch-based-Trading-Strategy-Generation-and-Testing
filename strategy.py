#!/usr/bin/env python3
"""
Experiment #10454: 1h 4h/1d Momentum Filter with Volume Spike
Hypothesis: 4h/1d trend direction combined with 1h momentum breakout and volume spike
provides high-probability entries with low trade frequency. Uses 4h EMA20 for trend,
1h Donchian breakout for entry, and volume confirmation to filter false signals.
Works in bull markets (breakouts above 4h EMA) and bear markets (breakdowns below 4h EMA).
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10454_1h_4h1d_momentum_filter_volume_spike_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
TREND_EMA_PERIOD = 20          # 4h EMA for trend filter
DONCHIAN_PERIOD = 15           # 1h Donchian for breakout
VOLUME_SPIKE_MULTIPLIER = 1.8  # Volume spike threshold
SESSION_START_HOUR = 8         # 8 UTC
SESSION_END_HOUR = 20          # 20 UTC
SIGNAL_SIZE = 0.20             # Position size (20%)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian_channels(high, low, period):
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= SESSION_START_HOUR) & (hours <= SESSION_END_HOUR)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = calculate_ema(df_4h['close'].values, TREND_EMA_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels for entry timing
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, DONCHIAN_PERIOD, 20)
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if 4h or 1d EMA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Trend filters: price above/both EMAs for long, below/both for short
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # 1h Donchian breakout for entry timing
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 4h/1d trend with volume
        long_entry = bullish_breakout and above_both_emas and volume_spike
        short_entry = bearish_breakout and below_both_emas and volume_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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