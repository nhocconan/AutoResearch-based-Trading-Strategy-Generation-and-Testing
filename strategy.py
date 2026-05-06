#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels with 4-hour EMA trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with price > 4h EMA50 and volume > 1.5x average
# Short when price breaks below Camarilla S3 level with price < 4h EMA50 and volume > 1.5x average
# Camarilla levels provide strong intraday support/resistance, EMA50 filters trend direction,
# Volume confirms breakout strength. Works in trending markets by capturing genuine breakouts.
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing.

name = "4h_1dCamarilla_R3S3_4hEMA50_Trend_Volume"
timeframe = "4h"
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
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4-hour EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
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
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Camarilla R3 with trend and volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Camarilla S3 with trend and volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (trend reversal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (trend reversal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals