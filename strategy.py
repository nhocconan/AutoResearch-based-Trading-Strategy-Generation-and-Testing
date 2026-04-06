#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points for direction and daily volume confirmation.
# Goes long when price closes above weekly R3 with above-average daily volume,
# short when closes below weekly S3 with above-average daily volume.
# Uses weekly pivot levels (R3/S3) as breakout/breakdown levels for strong moves.
# Weekly pivot provides institutional levels, volume confirms institutional participation.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Weekly pivots avoid whipsaw by requiring breaks of significant levels.

name = "exp_13827_6h_weekly_pivot_daily_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # weeks for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (R3, S3 levels)"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return r3, s3  # Return only R3 and S3 for breakout/breakdown

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (R3, S3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    r3_weekly, s3_weekly = calculate_weekly_pivot(high_weekly, low_weekly, close_weekly)
    
    # Align weekly pivot points to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_weekly)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_weekly)
    
    # Load daily data for volume confirmation ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily volume moving average
    volume_daily = df_daily['volume'].values
    volume_ma_daily = pd.Series(volume_daily).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align daily volume MA to 6h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_daily)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_ma_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation from daily data
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD)
        
        # Breakout/breakdown signals using weekly R3/S3
        long_signal = volume_ok and close[i] > r3_aligned[i]
        short_signal = volume_ok and close[i] < s3_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below weekly S3 (reversal signal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above weekly R3 (reversal signal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals