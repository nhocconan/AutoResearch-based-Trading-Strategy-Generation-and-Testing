#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals with weekly trend filter and volume confirmation
# Uses daily Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) with weekly trend filter
# to avoid counter-trend trades. Volume confirms institutional participation.
# Works in both bull (breakouts at R4/S4) and bear (reversals at R3/S3) markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13455_6h_camarilla_pivot_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily levels
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    # Resistance levels
    r3 = close + (range_hl * 1.1 / 4)
    r4 = close + (range_hl * 1.1 / 2)
    
    # Support levels
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Initialize arrays for Camarilla levels
    r3_daily = np.full(len(high_daily), np.nan)
    r4_daily = np.full(len(high_daily), np.nan)
    s3_daily = np.full(len(high_daily), np.nan)
    s4_daily = np.full(len(high_daily), np.nan)
    
    # Calculate Camarilla levels for each day
    for i in range(len(high_daily)):
        r3, r4, s3, s4 = calculate_camarilla(high_daily[i], low_daily[i], close_daily[i])
        r3_daily[i] = r3
        r4_daily[i] = r4
        s3_daily[i] = s3
        s4_daily[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Load weekly data for trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = calculate_ema(close_weekly, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_weekly_aligned[i]
        downtrend = close[i] < ema_weekly_aligned[i]
        
        # Camarilla-based signals
        # Long conditions:
        # 1. Mean reversion: price at S3 in uptrend (bounce from support in uptrend)
        # 2. Breakout: price breaks above R4 in uptrend (continuation)
        long_mean_reversion = volume_ok and uptrend and (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i])
        long_breakout = volume_ok and uptrend and (high[i] >= r4_aligned[i])
        
        # Short conditions:
        # 1. Mean reversion: price at R3 in downtrend (rejection at resistance in downtrend)
        # 2. Breakdown: price breaks below S4 in downtrend (continuation)
        short_mean_reversion = volume_ok and downtrend and (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i])
        short_breakdown = volume_ok and downtrend and (low[i] <= s4_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_mean_reversion or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_mean_reversion or short_breakdown:
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