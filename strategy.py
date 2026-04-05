#!/usr/bin/env python3
"""
Experiment #10231: 6h Camarilla Pivot Reversal with Daily Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3) act as strong reversal zones in the direction of the daily trend.
In trending markets (price above/below daily EMA50), price rejects at R3/S3 and continues the trend.
In ranging markets, reversals at R3/S3 still occur but with smaller size. Volume confirmation filters false signals.
Works in both bull (buy R3 bounces in uptrend) and bear (sell S3 rejections in downtrend).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10231_6h_camarilla_pivot_reversal_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLUME_SPIKE_MULTIPLIER = 1.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = pivot + (range_val * CAMARILLA_MULT * 1.1)
    s3 = pivot - (range_val * CAMARILLA_MULT * 1.1)
    return r3, s3

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
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter and pivot points
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DAILY_EMA_PERIOD, 2) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Calculate Camarilla levels for previous day (using previous bar's data)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_r3, camarilla_s3 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        # Volume spike confirmation
        volume_ma = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1] if i >= 20 else np.nan
        volume_spike = volume[i] > (volume_ma * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Reversal conditions at Camarilla levels
        # Long: price touches/bounces off S3 in uptrend
        long_setup = (close[i] <= camarilla_s3 * 1.002) and above_daily_ema  # Allow small penetration
        # Short: price touches/rejects at R3 in downtrend
        short_setup = (close[i] >= camarilla_r3 * 0.998) and below_daily_ema  # Allow small penetration
        
        # Entry conditions: reversal at Camarilla level with volume and trend alignment
        long_entry = long_setup and volume_spike
        short_entry = short_setup and volume_spike
        
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