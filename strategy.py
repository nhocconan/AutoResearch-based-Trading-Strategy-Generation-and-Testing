#!/usr/bin/env python3
"""
Experiment #10075: 6h Camarilla Pivot + Weekly Trend + Volume Spike
Hypothesis: Camarilla pivot levels from 1d provide high-probability reversal zones at R3/S3 (fade) 
and breakout continuation signals at R4/S4. Combined with weekly trend filter (EMA40) and volume
confirmation, this strategy captures both mean-reversion and trend continuation moves. 
Works in bull markets (buy R3/S3 bounces, break R4/S4) and bear markets (sell R3/S3 rallies, 
break R4/S4 down). Volume filters reduce false signals. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_10075_6h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
WEEKLY_EMA_PERIOD = 40
SIGNAL_SIZE = 0.25
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MIN_HOLD_BARS = 4  # Minimum 4 bars (~1 day) holding period

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    pivot = (high + low + close) / 3.0
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots for each daily bar
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    r4_d = np.full_like(daily_close, np.nan)
    r3_d = np.full_like(daily_close, np.nan)
    r2_d = np.full_like(daily_close, np.nan)
    r1_d = np.full_like(daily_close, np.nan)
    p_d = np.full_like(daily_close, np.nan)
    s1_d = np.full_like(daily_close, np.nan)
    s2_d = np.full_like(daily_close, np.nan)
    s3_d = np.full_like(daily_close, np.nan)
    s4_d = np.full_like(daily_close, np.nan)
    
    for i in range(len(daily_close)):
        r4, r3, r2, r1, p, s1, s2, s3, s4 = calculate_camarilla_pivot(
            daily_high[i], daily_low[i], daily_close[i]
        )
        r4_d[i] = r4
        r3_d[i] = r3
        r2_d[i] = r2
        r1_d[i] = r1
        p_d[i] = p
        s1_d[i] = s1
        s2_d[i] = s2
        s3_d[i] = s3
        s4_d[i] = s4
    
    # Align daily Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_daily, r4_d)
    r3_6h = align_htf_to_ltf(prices, df_daily, r3_d)
    r2_6h = align_htf_to_ltf(prices, df_daily, r2_d)
    r1_6h = align_htf_to_ltf(prices, df_daily, r1_d)
    p_6h = align_htf_to_ltf(prices, df_daily, p_d)
    s1_6h = align_htf_to_ltf(prices, df_daily, s1_d)
    s2_6h = align_htf_to_ltf(prices, df_daily, s2_d)
    s3_6h = align_htf_to_ltf(prices, df_daily, s3_d)
    s4_6h = align_htf_to_ltf(prices, df_daily, s4_d)
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_6h = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
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
    bars_held = 0
    
    # Start from warmup period
    start = max(20, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_held += 1
        
        # Skip if data not available
        if np.isnan(weekly_ema_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > weekly_ema_6h[i]
        below_weekly_ema = close[i] < weekly_ema_6h[i]
        
        # Camarilla levels
        r3 = r3_6h[i]
        s3 = s3_6h[i]
        r4 = r4_6h[i]
        s4 = s4_6h[i]
        
        # Fade at R3/S3 (mean reversion)
        fade_long = close[i] <= s3 and close[i] > s4 and volume_spike
        fade_short = close[i] >= r3 and close[i] < r4 and volume_spike
        
        # Breakout at R4/S4 (trend continuation)
        breakout_long = close[i] > r4 and above_weekly_ema and volume_spike
        breakout_short = close[i] < s4 and below_weekly_ema and volume_spike
        
        # Entry conditions
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
        # Generate signals
        if position == 0:
            if long_entry and bars_held >= MIN_HOLD_BARS:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            elif short_entry and bars_held >= MIN_HOLD_BARS:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_held = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals