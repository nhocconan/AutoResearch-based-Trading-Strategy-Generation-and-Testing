#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, take long positions when price breaks above Camarilla R3 from prior day
with 1-week uptrend (price above 200-period EMA) and volume confirmation (1.5x average).
Take short positions when price breaks below Camarilla S3 with 1-week downtrend.
Exit when price returns to the daily pivot point.
Uses weekly trend filter to avoid counter-trend trades and volume confirmation to reduce false breakouts.
Designed for low trade frequency (12-37/year) requiring multiple confluence factors.
Works in bull/bear via weekly trend filter and mean-reversion exit at pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's range)
    # Using prior day's data to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first day uses current
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    range_1d = prev_high_1d - prev_low_1d
    camarilla_mult = 1.1 / 6  # Camarilla multiplier
    
    # Resistance levels
    r3_1d = prev_close_1d + range_1d * camarilla_mult * 4
    r4_1d = prev_close_1d + range_1d * camarilla_mult * 5
    # Support levels
    s3_1d = prev_close_1d - range_1d * camarilla_mult * 4
    s4_1d = prev_close_1d - range_1d * camarilla_mult * 5
    # Pivot point
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    
    # === WEEKLY TREND FILTER (EMA 200) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    if len(close_1w) >= 200:
        ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    else:
        ema_200_1w = np.full_like(close_1w, np.nan)
    
    # === VOLUME AVERAGE (24-period for 12h = ~12 days) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align data to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below weekly EMA(200)
        price_above_ema = close[i] > ema_200_1w_aligned[i]
        price_below_ema = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > r3_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s3_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to daily pivot
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals