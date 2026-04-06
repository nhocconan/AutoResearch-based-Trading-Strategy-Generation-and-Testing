#!/usr/bin/env python3
"""
Experiment #12399: 6h Camarilla Pivot Fade with 12h Trend Filter
Hypothesis: Use Camarilla pivot levels from 1d (R3/S3 for fade, R4/S4 for breakout) with 12h trend filter.
Fade at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets.
Volume confirmation on entries. Works in bull via breakouts and in bear via fades/breakdowns.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12399_6h_camarilla_pivot_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # use previous day for Camarilla calculation
CAMARILLA_MULT = 1.1 / 12  # standard Camarilla multiplier
TREND_EMA_PERIOD = 34
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * CAMARILLA_MULT * 11/2)
    r3 = pp + (range_ * CAMARILLA_MULT * 5/2)
    r2 = pp + (range_ * CAMARILLA_MULT * 3/2)
    r1 = pp + (range_ * CAMARILLA_MULT * 1/2)
    s1 = pp - (range_ * CAMARILLA_MULT * 1/2)
    s2 = pp - (range_ * CAMARILLA_MULT * 3/2)
    s3 = pp - (range_ * CAMARILLA_MULT * 5/2)
    s4 = pp - (range_ * CAMARILLA_MULT * 11/2)
    
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for lookback)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    start = max(PIVOT_LOOKBACK, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1d or 12h data not available
        if np.isnan(r4_aligned[i]) or np.isnan(ema_12h_aligned[i]):
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
        
        # Trend filter (12h)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Price levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        # Fade conditions at R3/S3 (mean reversion in range)
        fade_long = volume_ok and close[i] <= r3_level and close[i] > s3_level
        fade_short = volume_ok and close[i] >= s3_level and close[i] < r3_level
        
        # Breakout conditions at R4/S4 (continuation in trend)
        breakout_long = volume_ok and uptrend_12h and close[i] >= r4_level
        breakout_short = volume_ok and downtrend_12h and close[i] <= s4_level
        
        # Entry conditions
        long_entry = (fade_long or breakout_long)
        short_entry = (fade_short or breakout_short)
        
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