#!/usr/bin/env python3
"""
Hypothesis: 4-hour ATR breakout with daily trend filter and volume confirmation.
Breakouts above/below ATR-based channels capture momentum moves, daily trend filter
ensures alignment with higher timeframe bias, and volume confirms institutional participation.
ATR-based stops limit drawdown. Designed to work in both bull and bear markets by
following the daily trend direction. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, length=14):
    """Average True Range"""
    if len(high) < length:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    atr[:length-1] = np.nan
    atr[length-1] = np.mean(tr[:length])
    for i in range(length, len(high)):
        atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for breakout channels
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    atr_14_1d = calculate_atr(d_high, d_low, d_close, 14)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(d_close).ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 4-period high/low for breakout channels
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR(14) + EMA(50) + volume avg(20) + 4-period channels
    start_idx = max(14, 50, 20, 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(high_4[i]) or np.isnan(low_4[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_aligned[i]
        
        # Current indicators
        atr_now = atr_14_aligned[i]
        ema_trend = ema_50_aligned[i]
        high_channel = high_4[i]
        low_channel = low_4[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: price relative to daily EMA50
        price_above_trend = price_now > ema_trend
        price_below_trend = price_now < ema_trend
        
        if position == 0:
            # Long breakout: price breaks above 4-period high + uptrend + volume
            if price_now > high_channel and price_above_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below 4-period low + downtrend + volume
            elif price_now < low_channel and price_below_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: breakdown below 4-period low or trend reversal
            if price_now < low_channel or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: breakout above 4-period high or trend reversal
            if price_now > high_channel or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ATRBreakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0