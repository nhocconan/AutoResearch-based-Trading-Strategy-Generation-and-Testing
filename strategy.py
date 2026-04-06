#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe, fading at R3/S3 (mean reversion)
# and breakout continuation at R4/S4 (trend following). Uses 1d trend filter (EMA50) to
# bias direction and volume confirmation for institutional participation. Works in both
# bull and bear markets: fades extremes in ranging markets, catches breakouts in trends.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13467_6h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = 1
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    """
    Calculate Camarilla pivot levels for given period
    Returns: (S1, S2, S3, S4, R1, R2, R3, R4)
    Based on previous period's high, low, close
    """
    # Typical price
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    s1 = close - (range_ * 1.0 / 6.0)
    s2 = close - (range_ * 2.0 / 6.0)
    s3 = close - (range_ * 3.0 / 6.0)
    s4 = close - (range_ * 4.0 / 6.0)
    r1 = close + (range_ * 1.0 / 6.0)
    r2 = close + (range_ * 2.0 / 6.0)
    r3 = close + (range_ * 3.0 / 6.0)
    r4 = close + (range_ * 4.0 / 6.0)
    
    return s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla pivots (using previous day's HLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Shift to get previous day's values for pivot calculation
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN as there's no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each day
    s1_1d = np.full_like(high_1d, np.nan)
    s2_1d = np.full_like(high_1d, np.nan)
    s3_1d = np.full_like(high_1d, np.nan)
    s4_1d = np.full_like(high_1d, np.nan)
    r1_1d = np.full_like(high_1d, np.nan)
    r2_1d = np.full_like(high_1d, np.nan)
    r3_1d = np.full_like(high_1d, np.nan)
    r4_1d = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        if not (np.isnan(high_1d_prev[i]) or np.isnan(low_1d_prev[i]) or np.isnan(close_1d_prev[i])):
            s1, s2, s3, s4, r1, r2, r3, r4 = calculate_camarilla(
                high_1d_prev[i], low_1d_prev[i], close_1d_prev[i]
            )
            s1_1d[i] = s1
            s2_1d[i] = s2
            s3_1d[i] = s3
            s4_1d[i] = s4
            r1_1d[i] = r1
            r2_1d[i] = r2
            r3_1d[i] = r3
            r4_1d[i] = r4
    
    # Align Camarilla levels to 6h timeframe
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(atr[i])):
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
        volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
        volume_ok = not np.isnan(volume_ma[i]) and volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Mean reversion at S3/R3
        mean_revert_long = (close[i] <= s3_1d_aligned[i]) and uptrend
        mean_revert_short = (close[i] >= r3_1d_aligned[i]) and downtrend
        
        # Breakout continuation at S4/R4
        breakout_long = (close[i] >= r4_1d_aligned[i]) and uptrend
        breakout_short = (close[i] <= s4_1d_aligned[i]) and downtrend
        
        # Combine signals with volume confirmation
        long_signal = volume_ok and (mean_revert_long or breakout_long)
        short_signal = volume_ok and (mean_revert_short or breakout_short)
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals