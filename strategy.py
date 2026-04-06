#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals with volume confirmation and 1d trend filter
# Works in bull/bear because: 
# - Camarilla levels provide precise reversal points (R3/S3 for fade, R4/S4 for breakout)
# - Volume filters ensure only significant moves trigger entries
# - 1d trend filter prevents counter-trend trading in strong markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "exp_12931_6h_camarilla_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
CAMARILLA_MULTIPLIER = 1.1
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    r4 = close_val + CAMARILLA_MULTIPLIER * range_val * 1.1 / 2
    r3 = close_val + CAMARILLA_MULTIPLIER * range_val * 1.1 / 4
    r2 = close_val + CAMARILLA_MULTIPLIER * range_val * 1.1 / 6
    r1 = close_val + CAMARILLA_MULTIPLIER * range_val * 1.1 / 12
    s1 = close_val - CAMARILLA_MULTIPLIER * range_val * 1.1 / 12
    s2 = close_val - CAMARILLA_MULTIPLIER * range_val * 1.1 / 6
    s3 = close_val - CAMARILLA_MULTIPLIER * range_val * 1.1 / 4
    s4 = close_val - CAMARILLA_MULTIPLIER * range_val * 1.1 / 2
    return r4, r3, r2, r1, close_val, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    r4_d = np.zeros(len(close_d))
    r3_d = np.zeros(len(close_d))
    r2_d = np.zeros(len(close_d))
    r1_d = np.zeros(len(close_d))
    pp_d = np.zeros(len(close_d))
    s1_d = np.zeros(len(close_d))
    s2_d = np.zeros(len(close_d))
    s3_d = np.zeros(len(close_d))
    s4_d = np.zeros(len(close_d))
    
    for i in range(len(close_d)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(high_d[i], low_d[i], close_d[i])
        r4_d[i] = r4
        r3_d[i] = r3
        r2_d[i] = r2
        r1_d[i] = r1
        pp_d[i] = pp
        s1_d[i] = s1
        s2_d[i] = s2
        s3_d[i] = s3
        s4_d[i] = s4
    
    # Align to 6h timeframe
    r4_d_aligned = align_htf_to_ltf(prices, df_daily, r4_d)
    r3_d_aligned = align_htf_to_ltf(prices, df_daily, r3_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_daily, r2_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_daily, r1_d)
    pp_d_aligned = align_htf_to_ltf(prices, df_daily, pp_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_daily, s1_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_daily, s2_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_daily, s3_d)
    s4_d_aligned = align_htf_to_ltf(prices, df_daily, s4_d)
    
    # Calculate daily trend (close vs 20 EMA)
    ema20_d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_d_aligned = align_htf_to_ltf(prices, df_daily, ema20_d)
    
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if (np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]) or 
            np.isnan(r4_d_aligned[i]) or np.isnan(s4_d_aligned[i]) or
            np.isnan(ema20_d_aligned[i])):
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
        
        # Determine 6h Camarilla levels from daily (using previous day's levels)
        # For fade at R3/S3: price rejects extreme levels
        # For breakout at R4/S4: price breaks through with volume
        
        fade_long = volume_ok and close[i] <= s3_d_aligned[i] and close[i] > s4_d_aligned[i]
        fade_short = volume_ok and close[i] >= r3_d_aligned[i] and close[i] < r4_d_aligned[i]
        breakout_long = volume_ok and close[i] >= r4_d_aligned[i] and close[i] > ema20_d_aligned[i]
        breakout_short = volume_ok and close[i] <= s4_d_aligned[i] and close[i] < ema20_d_aligned[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
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