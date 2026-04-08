#!/usr/bin/env python3
# 6h_weekly_camarilla_daily_trend_v1
# Hypothesis: Use weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe.
# Daily trend filter (EMA 50) ensures we trade in direction of higher timeframe.
# Mean reversion at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets.
# Volume confirmation filters out low-quality signals.
# Weekly pivots provide institutional levels that work in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_weekly_camarilla_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Based on previous week's high, low, close
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Camarilla formulas
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.25)
    # S3 = Close - ((High - Low) * 1.25)
    # S4 = Close - ((High - Low) * 1.5)
    camarilla_r4 = weekly_close + (weekly_high - weekly_low) * 1.5
    camarilla_r3 = weekly_close + (weekly_high - weekly_low) * 1.25
    camarilla_s3 = weekly_close - (weekly_high - weekly_low) * 1.25
    camarilla_s4 = weekly_close - (weekly_high - weekly_low) * 1.5
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s4)
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.3x 24-period average (4 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(24, 50) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below S3 or trend fails
            if close[i] < s3_aligned[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above R3 or trend fails
            if close[i] > r3_aligned[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion entries at S3/R3 (fade extreme)
            # Long at S3 when price is below S3 but above S4 (oversold bounce)
            if (close[i] <= s3_aligned[i] and close[i] > s4_aligned[i] and
                close[i] > ema_daily_aligned[i] and volume_filter):
                position = 1
                signals[i] = 0.25
            # Short at R3 when price is above R3 but below R4 (overbought rejection)
            elif (close[i] >= r3_aligned[i] and close[i] < r4_aligned[i] and
                  close[i] < ema_daily_aligned[i] and volume_filter):
                position = -1
                signals[i] = -0.25
            # Breakout entries at R4/S4 (continuation)
            # Long breakout when price breaks above R4 with uptrend
            elif (close[i] > r4_aligned[i] and close[i] > ema_daily_aligned[i] and
                  volume_filter):
                position = 1
                signals[i] = 0.25
            # Short breakdown when price breaks below S4 with downtrend
            elif (close[i] < s4_aligned[i] and close[i] < ema_daily_aligned[i] and
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals