#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_pivot_with_volume_filter
# Uses Camarilla pivot levels (support/resistance) from daily timeframe as key levels.
# Enters long when price touches S3 level with volume confirmation in 1h timeframe.
# Enters short when price touches R3 level with volume confirmation.
# Exits when price reaches opposite pivot level (S1/R1) or after 4 bars.
# Uses 4h trend filter (EMA50) to avoid counter-trend trades.
# Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods.
# Position size: 0.20 (20%) to manage drawdown in volatile markets.
# Target: 20-40 trades/year to minimize fee drag.

name = "1h_4h_1d_camarilla_pivot_with_volume_filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day values to avoid roll issues
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    # Support levels
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.3 * 20-period average (1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0  # hold for max 4 bars
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            bars_in_trade = 0
            continue
        
        # Reset bar counter when flat
        if position == 0:
            bars_in_trade = 0
        
        # Check session and volume filters
        if not session_filter[i] or not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            bars_in_trade += 1
            continue
        
        # Long conditions: price touches S3 and above 4h EMA50 (uptrend filter)
        if low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and ema_50_4h_aligned[i] < close[i]:
            if position != 1:
                position = 1
                bars_in_trade = 0
                signals[i] = 0.20
            else:
                signals[i] = 0.20
                bars_in_trade += 1
        # Short conditions: price touches R3 and below 4h EMA50 (downtrend filter)
        elif high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and ema_50_4h_aligned[i] > close[i]:
            if position != -1:
                position = -1
                bars_in_trade = 0
                signals[i] = -0.20
            else:
                signals[i] = -0.20
                bars_in_trade += 1
        # Exit conditions
        else:
            bars_in_trade += 1
            # Exit if reached opposite level or max time
            if position == 1 and (close[i] >= r1_aligned[i] or bars_in_trade >= 4):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] <= s1_aligned[i] or bars_in_trade >= 4):
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.20
                elif position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
    
    return signals