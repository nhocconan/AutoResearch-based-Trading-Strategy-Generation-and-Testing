#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with 1-day volume spike and 1-week trend filter.
# Camarilla levels act as support/resistance; price bouncing off these levels with volume
# confirms institutional interest. Weekly EMA ensures alignment with higher timeframe momentum.
# Works in bull markets (buying dips at support) and bear markets (selling rallies at resistance).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13262_12h_camarilla_pivot_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
EMA_PERIOD_WEEKLY = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    typical_price = (high + low + close) / 3
    # Range
    range_ = high - low
    # Camarilla levels
    L4 = close + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 2)
    L3 = close + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 4)
    L2 = close + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 6)
    L1 = close + (range_ * CAMARILLA_MULTIPLIER * 1.1 / 12)
    H1 = close - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 12)
    H2 = close - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 6)
    H3 = close - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 4)
    H4 = close - (range_ * CAMARILLA_MULTIPLIER * 1.1 / 2)
    return L4, L3, L2, L1, H1, H2, H3, H4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD_WEEKLY)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    L4 = np.full(len(close_1d), np.nan)
    L3 = np.full(len(close_1d), np.nan)
    L2 = np.full(len(close_1d), np.nan)
    L1 = np.full(len(close_1d), np.nan)
    H1 = np.full(len(close_1d), np.nan)
    H2 = np.full(len(close_1d), np.nan)
    H3 = np.full(len(close_1d), np.nan)
    H4 = np.full(len(close_1d), np.nan)
    
    # Calculate Camarilla levels for each day
    for i in range(len(close_1d)):
        L4[i], L3[i], L2[i], L1[i], H1[i], H2[i], H3[i], H4[i] = calculate_camarilla_levels(
            high_1d[i], low_1d[i], close_1d[i]
        )
    
    # Align Camarilla levels to 12h timeframe
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Calculate 12h indicators
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla level touches with volume and trend
        # Long setup: price touches L3 or L4 support in uptrend with volume
        long_setup = volume_ok and uptrend and (
            (low[i] <= L3_aligned[i] and close[i] > L3_aligned[i]) or  # Touch L3 and bounce
            (low[i] <= L4_aligned[i] and close[i] > L4_aligned[i])   # Touch L4 and bounce
        )
        
        # Short setup: price touches H3 or H4 resistance in downtrend with volume
        short_setup = volume_ok and downtrend and (
            (high[i] >= H3_aligned[i] and close[i] < H3_aligned[i]) or  # Touch H3 and reject
            (high[i] >= H4_aligned[i] and close[i] < H4_aligned[i])    # Touch H4 and reject
        )
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
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