#!/usr/bin/env python3
"""
Experiment #8127: 6-hour weekly pivot with daily trend filter and volume confirmation.
Hypothesis: Price bouncing off weekly pivot support/resistance with daily EMA alignment and volume confirmation captures mean-reversion bounces in ranging markets and breakout continuations in trending markets. Weekly pivot provides key institutional levels, daily EMA filters for trend alignment, and volume confirms institutional participation. Designed for 6h timeframe to target 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8127_6h_weekly_pivot_daily_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # days to calculate weekly pivot
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot(high, low, close):
    """Calculate standard pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot, r1, r2, r3, s1, s2, s3 = [], [], [], [], [], [], []
    for i in range(len(high_1w)):
        p, r1_, r2_, r3_, s1_, s2_, s3_ = calculate_pivot(high_1w[i], low_1w[i], close_1w[i])
        pivot.append(p)
        r1.append(r1_)
        r2.append(r2_)
        r3.append(r3_)
        s1.append(s1_)
        s2.append(s2_)
        s3.append(s3_)
    
    pivot = np.array(pivot)
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)
    s1 = np.array(s1)
    s2 = np.array(s2)
    s3 = np.array(s3)
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(pivot_6h[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        if i >= VOLUME_MA_PERIOD:
            volume_ma = np.mean(volume[i-VOLUME_MA_PERIOD:i])
            volume_confirmed = volume[i] > (volume_ma * VOLUME_THRESHOLD) if not np.isnan(volume_ma) else False
        else:
            volume_confirmed = False
        
        # Price action around weekly pivot levels
        near_s1 = abs(close[i] - s1_6h[i]) < (0.5 * atr[i]) if not np.isnan(s1_6h[i]) else False
        near_s2 = abs(close[i] - s2_6h[i]) < (0.5 * atr[i]) if not np.isnan(s2_6h[i]) else False
        near_s3 = abs(close[i] - s3_6h[i]) < (0.5 * atr[i]) if not np.isnan(s3_6h[i]) else False
        near_r1 = abs(close[i] - r1_6h[i]) < (0.5 * atr[i]) if not np.isnan(r1_6h[i]) else False
        near_r2 = abs(close[i] - r2_6h[i]) < (0.5 * atr[i]) if not np.isnan(r2_6h[i]) else False
        near_r3 = abs(close[i] - r3_6h[i]) < (0.5 * atr[i]) if not np.isnan(r3_6h[i]) else False
        
        # Entry conditions: bounce from support in bullish bias, bounce from resistance in bearish bias
        long_entry = bull_bias and (near_s1 or near_s2 or near_s3) and volume_confirmed
        short_entry = bear_bias and (near_r1 or near_r2 or near_r3) and volume_confirmed
        
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