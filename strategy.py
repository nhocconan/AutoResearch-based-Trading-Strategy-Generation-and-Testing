#!/usr/bin/env python3
"""
Experiment #10107: 6h 1D/1W Camarilla Pivot Reversal
Hypothesis: Camarilla pivot levels from daily and weekly timeframes provide high-probability reversal points. 
Price approaching daily S3/R3 with rejection (close back inside S3/R3) and weekly trend alignment offers mean reversion opportunities.
Works in both bull and bear markets as pivots adapt to volatility. Volume confirmation filters false signals.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10107_6h_camarilla_pivot_1d_1w_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    R4 = c + ((h - l) * 1.5000)
    R3 = c + ((h - l) * 1.1250)
    R2 = c + ((h - l) * 0.6250)
    R1 = c + ((h - l) * 0.3750)
    PP = (h + l + c) / 3
    S1 = c - ((h - l) * 0.3750)
    S2 = c - ((h - l) * 0.6250)
    S3 = c - ((h - l) * 1.1250)
    S4 = c - ((h - l) * 1.5000)
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels for daily and weekly
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Initialize arrays for Camarilla levels
    R4_d = np.full(len(close_d), np.nan)
    R3_d = np.full(len(close_d), np.nan)
    R2_d = np.full(len(close_d), np.nan)
    R1_d = np.full(len(close_d), np.nan)
    PP_d = np.full(len(close_d), np.nan)
    S1_d = np.full(len(close_d), np.nan)
    S2_d = np.full(len(close_d), np.nan)
    S3_d = np.full(len(close_d), np.nan)
    S4_d = np.full(len(close_d), np.nan)
    
    R4_w = np.full(len(close_w), np.nan)
    R3_w = np.full(len(close_w), np.nan)
    R2_w = np.full(len(close_w), np.nan)
    R1_w = np.full(len(close_w), np.nan)
    PP_w = np.full(len(close_w), np.nan)
    S1_w = np.full(len(close_w), np.nan)
    S2_w = np.full(len(close_w), np.nan)
    S3_w = np.full(len(close_w), np.nan)
    S4_w = np.full(len(close_w), np.nan)
    
    # Calculate pivots for each bar
    for i in range(len(close_d)):
        R4_d[i], R3_d[i], R2_d[i], R1_d[i], PP_d[i], S1_d[i], S2_d[i], S3_d[i], S4_d[i] = calculate_camarilla(high_d[i], low_d[i], close_d[i])
    
    for i in range(len(close_w)):
        R4_w[i], R3_w[i], R2_w[i], R1_w[i], PP_w[i], S1_w[i], S2_w[i], S3_w[i], S4_w[i] = calculate_camarilla(high_w[i], low_w[i], close_w[i])
    
    # Align Camarilla levels to 6h timeframe
    R3_d_aligned = align_htf_to_ltf(prices, df_daily, R3_d)
    S3_d_aligned = align_htf_to_ltf(prices, df_daily, S3_d)
    PP_d_aligned = align_htf_to_ltf(prices, df_daily, PP_d)
    R3_w_aligned = align_htf_to_ltf(prices, df_weekly, R3_w)
    S3_w_aligned = align_htf_to_ltf(prices, df_weekly, S3_w)
    PP_w_aligned = align_htf_to_ltf(prices, df_weekly, PP_w)
    
    # Weekly EMA for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, 50)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if np.isnan(R3_d_aligned[i]) or np.isnan(S3_d_aligned[i]) or np.isnan(weekly_ema_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Rejection conditions: price tests S3/R3 but closes back inside
        touched_S3_d = low[i] <= S3_d_aligned[i]
        rejected_S3_d = touched_S3_d and close[i] > S3_d_aligned[i]
        
        touched_R3_d = high[i] >= R3_d_aligned[i]
        rejected_R3_d = touched_R3_d and close[i] < R3_d_aligned[i]
        
        # Weekly trend filter
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Entry conditions: rejection of S3/R3 with weekly trend alignment and volume
        long_entry = rejected_S3_d and above_weekly_ema and volume_spike
        short_entry = rejected_R3_d and below_weekly_ema and volume_spike
        
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