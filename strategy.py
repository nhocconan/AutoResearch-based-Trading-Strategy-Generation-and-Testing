#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(10) breakout with daily volume confirmation and weekly EMA(50) trend filter.
# Goes long when price breaks above weekly Donchian upper band with above-average daily volume and price above weekly EMA50,
# short when breaks below weekly Donchian lower band with volume and price below weekly EMA50.
# Exits when price crosses the opposite Donchian band or when weekly EMA50 slope changes direction.
# Designed for 40-80 total trades over 4 years (10-20/year) to minimize fee drag while capturing major trends.

name = "exp_13838_1d_weekly_donchian10_weekly_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 10  # Weekly Donchian period
EMA_PERIOD = 50       # Weekly EMA period for trend
VOLUME_MA_PERIOD = 20 # Daily volume moving average
VOLUME_THRESHOLD = 1.5 # Volume must be 1.5x average
SIGNAL_SIZE = 0.25    # Position size (25% of capital)
ATR_PERIOD = 14       # ATR period for stop loss
ATR_STOP_MULTIPLIER = 2.5 # ATR multiplier for stop loss

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema_slope(ema_values, lookback=3):
    """Calculate EMA slope over lookback period"""
    ema_series = pd.Series(ema_values)
    # Calculate slope as (current - past) / lookback
    slope = (ema_series - ema_series.shift(lookback)) / lookback
    return slope.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels and EMA trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    weekly_upper, weekly_lower = calculate_donchian(high_weekly, low_weekly, DONCHIAN_PERIOD)
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema = calculate_ema(close_weekly, EMA_PERIOD)
    
    # Calculate weekly EMA slope for exit condition
    weekly_ema_slope = calculate_ema_slope(weekly_ema, lookback=3)
    
    # Align weekly data to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    weekly_ema_slope_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_slope)
    
    # Daily data for volume and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation (daily)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(weekly_ema_slope_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from weekly EMA
        above_ema = close[i] > weekly_ema_aligned[i]
        below_ema = close[i] < weekly_ema_aligned[i]
        
        # EMA slope conditions
        rising_slope = weekly_ema_slope_aligned[i] > 0
        falling_slope = weekly_ema_slope_aligned[i] < 0
        
        # Weekly Donchian breakout signals
        long_signal = volume_ok and above_ema and rising_slope and close[i] > weekly_upper_aligned[i]
        short_signal = volume_ok and below_ema and falling_slope and close[i] < weekly_lower_aligned[i]
        
        # Exit conditions: EMA slope change or price crosses opposite band
        exit_long = (not rising_slope) or (close[i] < weekly_lower_aligned[i])
        exit_short = (not falling_slope) or (close[i] > weekly_upper_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals