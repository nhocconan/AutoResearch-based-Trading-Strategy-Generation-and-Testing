#!/usr/bin/env python3
"""
Experiment #9771: 6h Donchian(20) Breakout + Daily Trend Filter + Volume Confirmation.
Hypothesis: Donchian breakouts on 6h timeframe, filtered by daily trend (price > SMA50 for long, < SMA50 for short) 
and confirmed by volume spikes, will capture trending moves while avoiding counter-trend whipsaws. 
This combines price channel breakouts (proven effective) with trend alignment and volume confirmation 
to reduce false signals. Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in both bull and bear markets by only taking longs in uptrends and shorts in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9771_6h_donchian_breakout_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
SMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load HTF data ONCE before loop (1d for daily trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA for trend filter
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean().values
    
    # Align 1d SMA to 6h timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, SMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(sma_1d_aligned[i]):
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
        
        # Trend filter: price above/below daily SMA
        uptrend = close[i] > sma_1d_aligned[i]
        downtrend = close[i] < sma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] >= highest_high[i]  # Using high to detect breakout
        breakout_down = low[i] <= lowest_low[i]   # Using low to detect breakdown
        
        # Entry conditions: breakout in direction of trend + volume confirmation
        long_entry = breakout_up and uptrend and volume_spike
        short_entry = breakout_down and downtrend and volume_spike
        
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