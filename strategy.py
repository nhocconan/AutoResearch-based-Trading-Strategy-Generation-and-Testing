#!/usr/bin/env python3
"""
Experiment #9530: 1d Donchian(20) Breakout + Weekly EMA + Volume Confirmation
Hypothesis: Donchian(20) breakouts on 1d timeframe, filtered by weekly EMA trend direction and volume spikes, 
provide high-probability trend continuation signals. Weekly EMA acts as a trend filter to avoid counter-trend trades.
Volume confirmation ensures breakouts have institutional participation. Targets 30-100 trades over 4 years.
Works in bull markets (breakouts above weekly EMA) and bear markets (breakouts below weekly EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9530_1d_donchian20_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50  # Weekly EMA for trend filter
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
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
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, EMA_PERIOD)
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: weekly EMA direction
        # For breakouts, we only trade in direction of weekly trend
        weekly_trend_up = close[i] > weekly_ema_aligned[i]
        weekly_trend_down = close[i] < weekly_ema_aligned[i]
        
        # Breakout conditions
        upper_breakout = high[i] > donchian_upper[i]  # Price breaks above upper band
        lower_breakout = low[i] < donchian_lower[i]   # Price breaks below lower band
        
        # Entry conditions: breakout in direction of weekly trend with volume
        long_entry = upper_breakout and weekly_trend_up and volume_spike
        short_entry = lower_breakout and weekly_trend_down and volume_spike
        
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