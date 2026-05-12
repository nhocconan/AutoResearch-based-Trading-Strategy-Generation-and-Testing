#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend
# Hypothesis: On 4h timeframe, use daily Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs) as entry triggers.
# Enter long when price breaks above R3 with volume confirmation and 12-hour EMA50 uptrend.
# Enter short when price breaks below S3 with volume confirmation and 12-hour EMA50 downtrend.
# Exit when price returns to the pivot point (mean) or reverses at opposite level.
# Uses 12h trend filter to avoid counter-trend trades, targeting 20-40 trades/year for low friction.
# Works in bull via R3/R4 breakouts and in bear via S3/S4 breakdowns with trend alignment.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Typical price (Pivot point)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = daily_pivot + daily_range * 1.25
    r4 = daily_pivot + daily_range * 1.5
    s3 = daily_pivot - daily_range * 1.25
    s4 = daily_pivot - daily_range * 1.5
    
    # Align daily levels to 4h timeframe (with 1-bar delay for completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    twelve_hour_close = df_12h['close'].values
    twelve_hour_ema50 = pd.Series(twelve_hour_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    twelve_hour_ema50_aligned = align_htf_to_ltf(prices, df_12h, twelve_hour_ema50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(twelve_hour_ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        r4_val = r4_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        pivot_val = pivot_aligned[i]
        twelve_hour_trend = twelve_hour_ema50_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and 12h uptrend
            if close[i] > r3_val and close[i] > twelve_hour_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and 12h downtrend
            elif close[i] < s3_val and close[i] < twelve_hour_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot or breaks below R4 (taking profits)
            if close[i] <= pivot_val or close[i] < r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot or breaks above S4 (taking profits)
            if close[i] >= pivot_val or close[i] > s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals