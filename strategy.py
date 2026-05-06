#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Camarilla pivot levels with 4-hour trend filter and volume confirmation
# Long when price breaks above R3 with 4h EMA50 > EMA100 and volume > 1.5x average
# Short when price breaks below S3 with 4h EMA50 < EMA100 and volume > 1.5x average
# Camarilla levels from 1d provide strong intraday support/resistance, 4h EMA filter ensures trend alignment,
# Volume confirms breakout strength. Works in bull/bear by trading with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_1dCamarilla_R3S3_4hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4-hour EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 100:
        return np.zeros(n)
    
    # EMA50 and EMA100 on 4h close
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100 = pd.Series(close_4h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 4h EMAs to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_100_aligned = align_htf_to_ltf(prices, df_4h, ema_100)
    
    # Trend filter: EMA50 > EMA100 for uptrend, EMA50 < EMA100 for downtrend
    uptrend = ema_50_aligned > ema_100_aligned
    downtrend = ema_50_aligned < ema_100_aligned
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_100_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with uptrend and volume confirmation
            if close[i] > r3_aligned[i] and uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with downtrend and volume confirmation
            elif close[i] < s3_aligned[i] and downtrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (mean reversion) or trend turns down
            if close[i] < s3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (mean reversion) or trend turns up
            if close[i] > r3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals