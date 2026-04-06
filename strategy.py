#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels (1w) with daily trend filter and volume confirmation.
# Long when: price crosses above weekly pivot (PP) with volume > 1.5x average and close above daily EMA50.
# Short when: price crosses below weekly pivot (PP) with volume > 1.5x average and close below daily EMA50.
# Exit on opposite pivot touch or ATR stop. Uses weekly pivot for structure and daily EMA for trend filter.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13815_6h_weekly_pivot_daily_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous week's data
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (PP, R1, S1, R2, S2, R3, S3)"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, s1, r2, s2, r3, s3

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load weekly data for pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pp, r1, s1, r2, s2, r3, s3 = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Shift by 1 to use only completed weekly bars (look-ahead prevention)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Align weekly pivot to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)  # Use S1 for stop/exit
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)  # Use R1 for stop/exit
    
    # Load daily data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    
    # Align daily EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for price action, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Stop loss: price below S1
            if close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            # ATR stop
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Stop loss: price above R1
            if close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            # ATR stop
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from daily EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Entry signals: price crossing weekly pivot with volume and trend alignment
        long_signal = volume_ok and above_ema and close[i] > pp_aligned[i] and close[i-1] <= pp_aligned[i-1]
        short_signal = volume_ok and below_ema and close[i] < pp_aligned[i] and close[i-1] >= pp_aligned[i-1]
        
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
            # Exit long on price below S1 (weekly support) or reverse signal
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on price above R1 (weekly resistance) or reverse signal
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals