#!/usr/bin/env python3
"""
Experiment #11571: 6h Weekly Pivot Reversal with Volume Confirmation
Hypothesis: Weekly pivot levels (R3/S3) act as strong support/resistance on 6b timeframe.
Price rejecting these levels with volume confirmation indicates institutional interest.
Works in bull (buying dips to S3) and bear (selling rallies to R3) by fading extremes.
Target: 50-150 trades over 4 years using strict pivot/volume confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11571_6h_weekly_pivot_rev_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # weeks for weekly pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MIN_HOLD_BARS = 4  # minimum 1-day hold (4x6h bars)

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points from prior week"""
    # Typical price
    pp = (high + low + close) / 3.0
    # Support and resistance levels
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, r2, r3, s1, s2, s3

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load weekly data ONCE before loop (using 1d as proxy for weekly aggregation)
    df_weekly = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot levels from prior week data
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use prior week's data (no look-ahead)
    pp, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(
        np.roll(weekly_high, 1), 
        np.roll(weekly_low, 1), 
        np.roll(weekly_close, 1)
    )
    # First value will be NaN due to roll, that's correct (no prior week)
    
    # Align weekly pivot levels to 6b timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Calculate 6b indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_held = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_held += 1
        
        # Skip if weekly data not available (first week)
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0 and bars_held >= MIN_HOLD_BARS:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
                if position != 0:
                    position = 0
                    bars_held = 0
            continue
        
        # Check stoploss and minimum hold
        if position != 0:
            if bars_held < MIN_HOLD_BARS:
                signals[i] = position * SIGNAL_SIZE
                continue
                
            # Stoploss: 2.5 * ATR
            if position == 1:  # long
                if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_held = 0
                    continue
            elif position == -1:  # short
                if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_held = 0
                    continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price action at pivot levels
        # Long setup: price near S3 with rejection (low touching/below S3 but close above)
        near_s3 = low[i] <= s3_aligned[i] * 1.002  # within 0.2% of S3
        rejecting_s3 = close[i] > s3_aligned[i] and close[i] > open[i]  # bullish close
        
        # Short setup: price near R3 with rejection (high touching/above R3 but close below)
        near_r3 = high[i] >= r3_aligned[i] * 0.998  # within 0.2% of R3
        rejecting_r3 = close[i] < r3_aligned[i] and close[i] < open[i]  # bearish close
        
        # Entry conditions
        long_entry = near_s3 and rejecting_s3 and volume_ok
        short_entry = near_r3 and rejecting_r3 and volume_ok
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_held = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_held = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals