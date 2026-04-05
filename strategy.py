#!/usr/bin/env python3
"""
Experiment #10035: 6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Fade at Camarilla R3/S3 levels and breakout at R4/S4 levels provides mean reversion in range
and trend continuation in trending markets. Volume confirmation filters false signals. Works in bull/bear
by adapting to market regime (range vs trend).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10035_6h_camarilla_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day for pivot calculation
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Camarilla formula: P = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1/2)
    # R3 = C + ((H - L) * 1.1/4)
    # S3 = C - ((H - L) * 1.1/4)
    # S4 = C - ((H - L) * 1.1/2)
    pivot = (high + low + close) / 3
    range_hl = high - low
    r4 = close + (range_hl * 1.1 / 2)
    r3 = close + (range_hl * 1.1 / 4)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    return r4, r3, s3, s4

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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    r4_daily, r3_daily, s3_daily, s4_daily = calculate_camarilla_levels(daily_high, daily_low, daily_close)
    
    # Align daily Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
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
    start = max(20, 1) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Price levels
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        price = close[i]
        
        # Determine market regime based on price relative to R3/S3
        # If price between S3 and R3 -> range (mean revert at S3/R3)
        # If price > R3 or < S3 -> trend (breakout at R4/S4)
        in_range = (price > s3) and (price < r3)
        in_uptrend = price >= r3
        in_downtrend = price <= s3
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if in_range:
            # Mean reversion: buy at S3, sell at R3
            long_entry = (price <= s3) and volume_spike
            short_entry = (price >= r3) and volume_spike
        elif in_uptrend:
            # Trend continuation: buy breakout at R4
            long_entry = (price >= r4) and volume_spike
        elif in_downtrend:
            # Trend continuation: sell breakdown at S4
            short_entry = (price <= s4) and volume_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</s>