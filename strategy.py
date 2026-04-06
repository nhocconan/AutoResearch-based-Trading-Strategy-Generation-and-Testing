#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Goes long when price breaks above weekly Donchian upper band with above-average volume and price above 1d EMA200,
# short when breaks below weekly Donchian lower band with volume and price below 1d EMA200.
# Weekly timeframe reduces trade frequency to target 30-100 total trades over 4 years.
# Weekly structure provides strong trend context, daily EMA filters alignment, volume confirms breakout validity.

name = "exp_13818_1d_weekly_donchian20_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels and EMA trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    weekly_upper, weekly_lower = calculate_donchian(high_weekly, low_weekly, WEEKLY_DONCHIAN_PERIOD)
    
    # Calculate 1d EMA for trend filter (using daily data from prices)
    close_daily = prices['close'].values
    ema_daily = calculate_ema(close_daily, EMA_PERIOD)
    
    # Daily data for ATR and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align weekly Donchian levels to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or 
            np.isnan(ema_daily[i]) or np.isnan(volume_ma[i])):
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
        
        # Trend direction from daily EMA
        above_ema = close[i] > ema_daily[i]
        below_ema = close[i] < ema_daily[i]
        
        # Weekly Donchian breakout signals
        long_signal = volume_ok and above_ema and close[i] > weekly_upper_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < weekly_lower_aligned[i]
        
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
            # Exit long on close below weekly Donchian lower band
            if close[i] < weekly_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above weekly Donchian upper band
            if close[i] > weekly_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals