#!/usr/bin/env python3
"""
Experiment #10291: 6h 123 Reversal + Volume Spike + Daily Trend
Hypothesis: The 123 reversal pattern (higher low in downtrend, lower high in uptrend) 
combined with volume spikes and daily trend filter provides high-probability reversal 
entries. Works in both bull and bear markets by capturing exhaustion moves. 
Volume confirms institutional participation. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10291_6h_123_reversal_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
LOOKBACK_PERIOD = 10
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(LOOKBACK_PERIOD + 2, 20) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # 123 Reversal pattern detection
        # Need at least 3 bars of lookback
        if i >= LOOKBACK_PERIOD + 2:
            # Get recent lows and highs
            recent_lows = low[i-LOOKBACK_PERIOD:i+1]
            recent_highs = high[i-LOOKBACK_PERIOD:i+1]
            
            # Find lowest low and highest high in lookback period
            lowest_low = np.min(recent_lows)
            highest_high = np.max(recent_highs)
            
            # Current bar indices relative to lookback start
            current_low = low[i]
            current_high = high[i]
            
            # Higher Low pattern (for long): 
            # 1. Price made a higher low than the lowest low in lookback
            # 2. Current close is above the prior bar's close (showing strength)
            # 3. Volume spike confirms
            higher_low = (current_low > lowest_low) and (close[i] > close[i-1])
            
            # Lower High pattern (for short):
            # 1. Price made a lower high than the highest high in lookback
            # 2. Current close is below the prior bar's close (showing weakness)
            # 3. Volume spike confirms
            lower_high = (current_high < highest_high) and (close[i] < close[i-1])
            
            # Entry conditions
            long_entry = higher_low and above_daily_ema and volume_spike
            short_entry = lower_high and below_daily_ema and volume_spike
            
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
        else:
            # Not enough data for pattern detection
            signals[i] = 0.0
    
    return signals