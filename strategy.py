#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal strategy with daily volatility filter.
# Uses daily Camarilla levels (S3/S4 for long, R3/R4 for short) to identify extreme intraday reversals.
# Volatility filter ensures trades only occur during sufficient market movement.
# Works in ranging markets (mean reversion at extremes) and trending markets (pullbacks to pivot levels).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13545_12h_camarilla1d_vol_filter_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Multiplier for Camarilla width calculation
VOLATILITY_LOOKBACK = 24  # 24 * 12h = 12 days for volatility assessment
VOLATILITY_THRESHOLD = 0.5  # Minimum ATR as % of price to ensure sufficient movement
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: range = (high - low) * 1.1
    # S3 = close - (high - low) * 1.1 / 6
    # S4 = close - (high - low) * 1.1 / 2
    # R3 = close + (high - low) * 1.1 / 6
    # R4 = close + (high - low) * 1.1 / 2
    range_1d = (high_1d - low_1d) * CAMARILLA_MULT
    s3 = close_1d - (range_1d / 6)
    s4 = close_1d - (range_1d / 2)
    r3 = close_1d + (range_1d / 6)
    r4 = close_1d + (range_1d / 2)
    
    # Align Camarilla levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for volatility filter and stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volatility filter: ensure sufficient price movement
    volatility_ratio = atr / close
    volatility_ok = volatility_ratio > VOLATILITY_THRESHOLD / 100  # Convert percentage to decimal
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(len(df_1d), ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]):
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
        
        # Volatility filter
        if not volatility_ok[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Camarilla reversal signals
        # Long when price touches or goes below S3/S4 (extreme low)
        # Short when price touches or goes above R3/R4 (extreme high)
        touch_s3 = low[i] <= s3_aligned[i]
        touch_s4 = low[i] <= s4_aligned[i]
        touch_r3 = high[i] >= r3_aligned[i]
        touch_r4 = high[i] >= r4_aligned[i]
        
        # Generate signals
        if position == 0:
            if touch_s3 or touch_s4:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif touch_r3 or touch_r4:
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