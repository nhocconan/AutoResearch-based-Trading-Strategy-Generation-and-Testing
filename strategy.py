#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day Camarilla pivot levels.
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought).
# Camarilla pivots from 1-day provide S1-S4 and R1-R4 levels for mean reversion:
#   - Buy near S3/S4 when Williams %R oversold (< -80) with price > S3
#   - Sell near R3/R4 when Williams %R overbought (> -20) with price < R3
# Works in ranging markets (mean reversion at extremes) and can capture breaks during trends.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13515_6h_williamsr_1d_camarilla_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * 1.1 / 2)
    r3 = pp + (range_ * 1.1 / 4)
    r2 = pp + (range_ * 1.1 / 6)
    r1 = pp + (range_ * 1.1 / 12)
    s1 = pp - (range_ * 1.1 / 12)
    s2 = pp - (range_ * 1.1 / 6)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h Williams %R
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Williams %R or Camarilla levels not available
        if np.isnan(williams_r[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        
        # Mean reversion signals using Williams %R and Camarilla levels
        # Long setup: Williams %R oversold AND price above S3 (support holding)
        long_setup = (williams_r[i] < WILLIAMS_OVERSOLD) and (close[i] > s3_1d_aligned[i])
        
        # Short setup: Williams %R overbought AND price below R3 (resistance holding)
        short_setup = (williams_r[i] > WILLIAMS_OVERBOUGHT) and (close[i] < r3_1d_aligned[i])
        
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