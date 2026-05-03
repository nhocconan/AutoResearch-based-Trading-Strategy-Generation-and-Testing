#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation
# Uses weekly EMA50 to determine primary trend direction, then trades breakouts of
# daily Camarilla R3/S3 levels in the direction of the weekly trend with volume spike confirmation.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Weekly trend filter helps avoid counter-trend trades in both bull and bear markets.

name = "6h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    # Avoid look-ahead: use previous day's data only
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    daily_range = prev_high - prev_low
    
    camarilla_r3 = pivot_point + (daily_range * 1.1 / 4.0)
    camarilla_s3 = pivot_point - (daily_range * 1.1 / 4.0)
    camarilla_r4 = pivot_point + (daily_range * 1.1 / 2.0)
    camarilla_s4 = pivot_point - (daily_range * 1.1 / 2.0)
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate daily volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ema_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_weekly_uptrend = close[i] > weekly_trend_aligned[i]
        is_weekly_downtrend = close[i] < weekly_trend_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in weekly uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and is_weekly_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in weekly downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and is_weekly_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla S3 (mean reversion) or weekly trend changes
            if close[i] < camarilla_s3_aligned[i] or not is_weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla R3 (mean reversion) or weekly trend changes
            if close[i] > camarilla_r3_aligned[i] or not is_weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals