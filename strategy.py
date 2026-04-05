#!/usr/bin/env python3
"""
Experiment #9714: 1h Donchian Breakout + Volume Spike + 4h Trend Filter.
Hypothesis: 1h price breaking above/below 4h Donchian channel (20-period) with volume confirmation 
captures institutional breakouts. Trend filter (4h EMA50) ensures alignment with higher timeframe 
direction, reducing whipsaws. Targets 60-150 total trades over 4 years (15-37/year) by using 
strict entry conditions and session filter (08-20 UTC). Works in bull markets via breakouts and 
bear markets via breakdowns, with volume filter ensuring participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9714_1h_donchian_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_PERIOD = 50
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
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
    
    # Load HTF data ONCE before loop (4h for trend and Donchian)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channel (20-period)
    donch_upper_4h, donch_lower_4h = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    
    # 4h EMA50 for trend filter
    ema_4h = calculate_ema(close_4h, EMA_PERIOD)
    
    # Align 4h indicators to 1h timeframe
    donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
    donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i]) or np.isnan(ema_4h_aligned[i]):
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
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        breakout_up = close[i] > donch_upper_4h_aligned[i]
        breakout_down = close[i] < donch_lower_4h_aligned[i]
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and uptrend and in_session
        short_entry = breakout_down and volume_spike and downtrend and in_session
        
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