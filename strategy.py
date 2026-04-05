#!/usr/bin/env python3
"""
exp_7411_6d_pivot3levels_volatility_breakout_v1
Hypothesis: 6-hour pivot (3-level) volatility breakout with 1-day trend filter.
Uses pivot points from previous day + volatility expansion (volume > 2x MA) to capture breakouts.
Only trades in direction of 1-day EMA(50) to avoid counter-trend whipsaws.
Designed for low trade frequency (target: 50-150 total over 4 years) with 6h timeframe.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7411_6d_pivot3levels_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's data
VOL_MA_PERIOD = 20
VOL_BREAKOUT_THRESHOLD = 2.0
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # Max 3 days

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, S1, R2, S2, R3, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate pivot points from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Calculate pivot levels for each day
    pivot_vals = np.full(len(high_1d), np.nan)
    r1_vals = np.full(len(high_1d), np.nan)
    s1_vals = np.full(len(high_1d), np.nan)
    r2_vals = np.full(len(high_1d), np.nan)
    s2_vals = np.full(len(high_1d), np.nan)
    r3_vals = np.full(len(high_1d), np.nan)
    s3_vals = np.full(len(high_1d), np.nan)
    
    for i in range(len(high_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d_for_pivot[i])):
            p, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(high_1d[i], low_1d[i], close_1d_for_pivot[i])
            pivot_vals[i] = p
            r1_vals[i] = r1
            s1_vals[i] = s1
            r2_vals[i] = r2
            s2_vals[i] = s2
            r3_vals[i] = r3
            s3_vals[i] = s3
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for breakout confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume breakout confirmation
        vol_breakout = volume[i] > vol_ma[i] * VOL_BREAKOUT_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Breakout conditions using pivot levels
        # Long: break above R1 with volume, only in uptrend
        breakout_long = above_ema and vol_breakout and (close[i] > r1_aligned[i])
        # Short: break below S1 with volume, only in downtrend
        breakout_short = below_ema and vol_breakout and (close[i] < s1_aligned[i])
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals