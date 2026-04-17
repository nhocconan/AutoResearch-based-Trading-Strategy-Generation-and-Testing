#!/usr/bin/env python3
"""
4h_ADX_TRIX_Signal_V1
Strategy: 4h TRIX signal cross + ADX trend filter + volume confirmation.
Long: TRIX crosses above signal line + ADX > 25 + volume > 1.5x 20-bar avg
Short: TRIX crosses below signal line + ADX > 25 + volume > 1.5x 20-bar avg
Exit: TRIX crosses back through signal line (mean reversion within trend)
Position size: 0.25
Uses TRIX(12) for momentum, ADX(14) for trend strength, volume for confirmation.
Designed to work in both bull and bear markets by requiring trending conditions (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) on close prices
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    # Signal line = EMA(TRIX, 9)
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean().values
    trix = 100 * (pd.Series(ema3).ewm(span=12, adjust=False).mean().values - 
                  np.roll(pd.Series(ema3).ewm(span=12, adjust=False).mean().values, 1)) / \
           np.roll(pd.Series(ema3).ewm(span=12, adjust=False).mean().values, 1)
    trix[0] = 0  # first value has no previous
    
    # Signal line: EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False).mean().values
    
    # Calculate ADX(14)
    high = prices['high'].values
    low = prices['low'].values
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smooth(dx, 14)
    
    # Volume confirmation: 20-period moving average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to ensure proper timing
    trix_aligned = align_htf_to_ltf(prices, prices, trix)
    signal_line_aligned = align_htf_to_ltf(prices, prices, signal_line)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    volume_ma20_aligned = align_htf_to_ltf(prices, prices, volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(40, n):  # warmup for TRIX and ADX
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20_aligned[i])
        trend_filter = adx_aligned[i] > 25  # trending market
        
        # TRIX cross signals
        trix_cross_up = trix_aligned[i] > signal_line_aligned[i] and trix_aligned[i-1] <= signal_line_aligned[i-1]
        trix_cross_down = trix_aligned[i] < signal_line_aligned[i] and trix_aligned[i-1] >= signal_line_aligned[i-1]
        
        if position == 0:
            # Long: TRIX crosses above signal + volume + trend
            if trix_cross_up and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal + volume + trend
            elif trix_cross_down and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses back below signal line
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses back above signal line
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_TRIX_Signal_V1"
timeframe = "4h"
leverage = 1.0