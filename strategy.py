#!/usr/bin/env python3
"""
Experiment #10215: 6h Elder Ray + Volume Spike + 1d Trend
Hypothesis: Elder Ray (Bull/Bear Power) identifies bull/bear strength; entering on power expansions with volume spikes in the direction of the daily trend filters false signals. Works in bull markets (bull power expansion) and bear markets (bear power expansion). Volume confirms institutional participation. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10215_6h_elder_ray_volume_spike_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
VOLUME_SPIKE_MULTIPLIER = 2.0
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    ema_close = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # Smooth the power values
    bull_power_smooth = pd.Series(bull_power).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD * 2, DAILY_EMA_PERIOD, 20) + 1
    
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
        
        # Elder Ray signals: expanding power in direction of trend
        bull_power_expanding = bull_power_smooth[i] > bull_power_smooth[i-1] if i > 0 else False
        bear_power_expanding = bear_power_smooth[i] < bear_power_smooth[i-1] if i > 0 else False  # more negative = expanding bear power
        
        # Entry conditions: power expansion with volume in direction of daily trend
        long_entry = bull_power_expanding and above_daily_ema and volume_spike
        short_entry = bear_power_expanding and below_daily_ema and volume_spike
        
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