#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with daily volume confirmation and weekly ADX trend filter.
# Goes long when price breaks above weekly Donchian upper band with above-average daily volume and weekly ADX > 25,
# short when breaks below weekly Donchian lower band with volume and weekly ADX > 25.
# Uses ATR-based stop loss to manage risk.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drain.
# Weekly Donchian provides strong structural breaks, daily volume confirms institutional interest,
# weekly ADX filters for trending environments to avoid whipsaws in ranging markets.

name = "exp_13824_1d_weekly_donchian20_daily_vol_adx_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_DONCHIAN_PERIOD = 20
DAILY_VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
WEEKLY_ADX_PERIOD = 14
WEEKLY_ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.zeros(len(high))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx) | np.isinf(dx)] = 0
    
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

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
    
    # Load weekly data for Donchian and ADX filters ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_upper, weekly_lower = calculate_donchian(weekly_high, weekly_low, WEEKLY_DONCHIAN_PERIOD)
    
    # Calculate weekly ADX for trend filter
    weekly_close = df_weekly['close'].values
    weekly_adx = calculate_adx(weekly_high, weekly_low, weekly_close, WEEKLY_ADX_PERIOD)
    
    # Align weekly indicators to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    weekly_adx_aligned = align_htf_to_ltf(prices, df_weekly, weekly_adx)
    
    # Daily data for price, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily ATR for stop loss
    daily_atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Daily volume confirmation
    volume_ma = pd.Series(volume).rolling(window=DAILY_VOLUME_MA_PERIOD, min_periods=DAILY_VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, WEEKLY_ADX_PERIOD, DAILY_VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or 
            np.isnan(weekly_adx_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(daily_atr[i])):
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
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = weekly_adx_aligned[i] > WEEKLY_ADX_THRESHOLD
        
        # Weekly Donchian breakout signals
        long_signal = volume_ok and trending and close[i] > weekly_upper_aligned[i]
        short_signal = volume_ok and trending and close[i] < weekly_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * daily_atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * daily_atr[i])
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