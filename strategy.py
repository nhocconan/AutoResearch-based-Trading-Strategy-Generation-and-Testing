#!/usr/bin/env python3
"""
Experiment #8455: 6h Camarilla pivot with volume confirmation and 1w trend filter.
Hypothesis: Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) on 1d chart
provide institutional support/resistance. Combined with 1w trend filter to avoid
counter-trend trades and volume confirmation for institutional participation.
Targets 50-150 total trades over 4 years (12-37/year) to balance frequency and edge.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8455_6h_camarilla_pivot_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily pivot (uses previous day's OHLC)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
TREND_PERIOD = 50

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
    r1 = close + (range_hl * 1.1 / 12)
    r2 = close + (range_hl * 1.1 / 6)
    r3 = close + (range_hl * 1.1 / 4)
    r4 = close + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_hl * 1.1 / 12)
    s2 = close - (range_hl * 1.1 / 6)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 1w EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1w > ema_1w, 1, 
                     np.where(close_1w < ema_1w, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    r3_1d = np.full(len(df_1d), np.nan)
    r4_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    # Calculate Camarilla for each day (starting from index 1)
    for i in range(1, len(df_1d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(r3_1d_aligned[i]):
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
        
        # Determine market bias from 1w EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1w price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1w price below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based conditions
        # Fade at R3/S3 (counter-trend pullback to strong level)
        fade_long = close[i] <= s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]
        fade_short = close[i] >= r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]
        
        # Breakout continuation at R4/S4 (strong break of key level)
        breakout_long = close[i] > r4_1d_aligned[i]
        breakout_short = close[i] < s4_1d_aligned[i]
        
        # Entry conditions: fade in ranging, breakout in trending
        # Use volume confirmation for both
        long_entry = volume_confirmed and (
            (fade_long and not bull_bias) or  # Fade long when not strongly bullish
            (breakout_long and bull_bias)     # Breakout long when bullish
        )
        
        short_entry = volume_confirmed and (
            (fade_short and not bear_bias) or  # Fade short when not strongly bearish
            (breakout_short and bear_bias)     # Breakout short when bearish
        )
        
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