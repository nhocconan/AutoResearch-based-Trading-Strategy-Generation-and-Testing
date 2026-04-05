#!/usr/bin/env python3
"""
Experiment #11607: 6h Weekly Pivot Reversal with Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Price rejecting at weekly R3/S3 with volume confirmation indicates reversal. In bull markets, bounces from weekly S1/S2; in bear markets, rejections from weekly R1/R2. Weekly timeframe filters noise, 6h provides timely entries. Target: 80-160 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11607_6h_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # bars to confirm pivot rejection
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_pivots(high, low, close):
    """Calculate weekly pivot points (standard formula)"""
    pp = (high + low + close) / 3.0
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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp, r1, r2, r3, s1, s2, s3 = calculate_pivots(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivots to 6h
    pp_6h = align_htf_to_ltf(prices, df_weekly, pp)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pp_6h[i]):
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
        
        # Price position relative to weekly pivots
        near_r3 = abs(close[i] - r3_6h[i]) < (0.5 * atr[i]) if not np.isnan(r3_6h[i]) else False
        near_s3 = abs(close[i] - s3_6h[i]) < (0.5 * atr[i]) if not np.isnan(s3_6h[i]) else False
        near_r2 = abs(close[i] - r2_6h[i]) < (0.5 * atr[i]) if not np.isnan(r2_6h[i]) else False
        near_s2 = abs(close[i] - s2_6h[i]) < (0.5 * atr[i]) if not np.isnan(s2_6h[i]) else False
        near_r1 = abs(close[i] - r1_6h[i]) < (0.5 * atr[i]) if not np.isnan(r1_6h[i]) else False
        near_s1 = abs(close[i] - s1_6h[i]) < (0.5 * atr[i]) if not np.isnan(s1_6h[i]) else False
        
        # Rejection signals (price moving away from pivot after touch)
        rejected_at_r3 = near_r3 and close[i] < r3_6h[i] and low[i] < r3_6h[i]  # touched R3, now below
        rejected_at_s3 = near_s3 and close[i] > s3_6h[i] and high[i] > s3_6h[i]  # touched S3, now above
        rejected_at_r2 = near_r2 and close[i] < r2_6h[i] and low[i] < r2_6h[i]  # touched R2, now below
        rejected_at_s2 = near_s2 and close[i] > s2_6h[i] and high[i] > s2_6h[i]  # touched S2, now above
        rejected_at_r1 = near_r1 and close[i] < r1_6h[i] and low[i] < r1_6h[i]  # touched R1, now below
        rejected_at_s1 = near_s1 and close[i] > s1_6h[i] and high[i] > s1_6h[i]  # touched S1, now above
        
        # Entry conditions
        long_entry = (rejected_at_s3 or rejected_at_s2 or rejected_at_s1) and volume_ok
        short_entry = (rejected_at_r3 or rejected_at_r2 or rejected_at_r1) and volume_ok
        
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