#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot with daily volume confirmation and weekly trend filter.
# Uses daily Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) with
# volume confirmation to avoid false signals. Weekly EMA filter ensures trades align
# with higher timeframe trend. Designed to work in both bull (breakouts) and bear
# (mean reversion) markets by adapting to volatility regimes. Target: 50-150 total
# trades over 4 years (12-37/year).

name = "exp_13495_6h_camarilla1d_vol_wk_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
WEEKLY_EMA_PERIOD = 21

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
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close
    c = (high + low + close) / 3
    r4 = c + ((high - low) * 1.5)
    r3 = c + ((high - low) * 1.1)
    s3 = c - ((high - low) * 1.1)
    s4 = c - ((high - low) * 1.5)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d = np.zeros(len(df_1d))
    r4_1d = np.zeros(len(df_1d))
    s3_1d = np.zeros(len(df_1d))
    s4_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Load weekly EMA for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla-based signals
        # Mean reversion at S3/R3 (buy at support, sell at resistance)
        mean_rev_long = volume_ok and (low[i] <= s3_1d_aligned[i]) and downtrend
        mean_rev_short = volume_ok and (high[i] >= r3_1d_aligned[i]) and uptrend
        
        # Breakout continuation at S4/R4 (breakdown/breakout with trend)
        breakout_long = volume_ok and (high[i] >= r4_1d_aligned[i]) and uptrend
        breakout_short = volume_ok and (low[i] <= s4_1d_aligned[i]) and downtrend
        
        # Generate signals
        if position == 0:
            if mean_rev_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_rev_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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