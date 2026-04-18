#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter.
In low volatility (BB width < 20th percentile), price builds energy. Breakout above upper band (long) or below lower band (short) captures the move.
Weekly trend filter (price > weekly EMA20 for longs, < for shorts) ensures we trade with the higher timeframe trend, reducing false breakouts in chop.
Designed for 15-25 trades/year to minimize fee drag. Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands. Returns: upper, lower, width"""
    if len(close) < period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = upper - lower
    return upper, lower, width

def calculate_ema(close, period):
    """Calculate EMA."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_weekly = calculate_ema(close_weekly, 20)
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Calculate Bollinger Bands on 6h data
    upper, lower, width = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    width_percentile = np.full(n, np.nan)
    for i in range(20, n):
        if not np.isnan(width[i]):
            past_widths = width[i-20:i]
            valid_widths = past_widths[~np.isnan(past_widths)]
            if len(valid_widths) > 0:
                percentile = (np.sum(valid_widths < width[i]) / len(valid_widths)) * 100
                width_percentile[i] = percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need BB calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(width_percentile[i]) or np.isnan(ema20_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        squeeze = width_percentile[i] < 20
        
        if position == 0 and squeeze:
            # Long: breakout above upper band with weekly uptrend
            if close[i] > upper[i] and close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band with weekly downtrend
            elif close[i] < lower[i] and close[i] < ema20_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band (SMA20) or opposite breakdown
            sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
            if not np.isnan(sma20) and close[i] < sma20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band (SMA20) or opposite breakout
            sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
            if not np.isnan(sma20) and close[i] > sma20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_Squeeze_WeeklyTrend"
timeframe = "6h"
leverage = 1.0