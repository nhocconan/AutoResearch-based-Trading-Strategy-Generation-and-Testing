#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Weekly Donchian(20) breakout with volume confirmation and 1d EMA200 filter.
# Goes long when price breaks above weekly Donchian upper band with volume > 1.5x 20-day average and price above daily EMA200.
# Goes short when price breaks below weekly Donchian lower band with volume confirmation and price below daily EMA200.
# Exits when price crosses back below/above the weekly Donchian opposite band.
# Uses weekly timeframe for structure to reduce trade frequency, daily EMA for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13818_1d_weekly_donchian20_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_upper, weekly_lower = calculate_donchian(weekly_high, weekly_low, WEEKLY_DONCHIAN_PERIOD)
    
    # Align weekly Donchian to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    
    # Calculate daily EMA for trend filter
    close = prices['close'].values
    ema = calculate_ema(close, EMA_PERIOD)
    
    # Volume confirmation
    volume = prices['volume'].values
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or np.isnan(ema[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from daily EMA
        above_ema = close[i] > ema[i]
        below_ema = close[i] < ema[i]
        
        # Weekly Donchian breakout signals
        long_signal = volume_ok and above_ema and close[i] > weekly_upper_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < weekly_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price crosses below weekly lower band
            if close[i] < weekly_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price crosses above weekly upper band
            if close[i] > weekly_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals