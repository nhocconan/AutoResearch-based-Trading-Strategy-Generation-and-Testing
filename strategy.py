#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter (1w EMA200) and volume confirmation.
# Camarilla levels provide institutional-grade support/resistance from prior day's range.
# Weekly EMA200 filters for long-term trend alignment to avoid counter-trend trades.
# Volume confirmation ensures breakout conviction.
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Works in bull markets (long above weekly EMA200) and bear markets (short below weekly EMA200).
name = "6h_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
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
    
    # Get daily data for Camarilla calculation (daily OHLC from previous day)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # R4 = close + 1.5 * (high - low), R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low), S4 = close - 1.5 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate using previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_1d = prev_high - prev_low
    r3 = prev_close + 1.0 * range_1d
    s3 = prev_close - 1.0 * range_1d
    
    # Get weekly data for trend filter (1w EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 24-period average volume for confirmation (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 00-23 UTC (6h bars span multiple sessions, use permissive filter)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        # Permissive session filter - allow all hours since 6h bars span sessions
        # But avoid extremely low volume periods (00-06 UTC typically quieter)
        in_session = not (0 <= hour <= 5)  # Avoid 00-06 UTC
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # Long: price breaks above R3 AND above weekly EMA200 AND volume confirmation
            long_breakout = close[i] > r3_aligned[i]
            above_weekly_trend = close[i] > ema200_1w_aligned[i]
            if vol_confirm and long_breakout and above_weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below weekly EMA200 AND volume confirmation
            elif vol_confirm and close[i] < s3_aligned[i] and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S3 OR falls below weekly EMA200
            exit_condition = close[i] < s3_aligned[i] or close[i] < ema200_1w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R3 OR rises above weekly EMA200
            exit_condition = close[i] > r3_aligned[i] or close[i] > ema200_1w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals