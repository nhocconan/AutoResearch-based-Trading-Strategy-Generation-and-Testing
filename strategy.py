#!/usr/bin/env python3
# 6h_Weekly_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaks above weekly R3 (resistance 3) or below weekly S3 (support 3) 
# with daily trend alignment (EMA50) and volume confirmation. 
# Weekly pivots define strong support/resistance; breaks indicate momentum continuation.
# Works in bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend) markets.
# Low frequency due to strict weekly pivot levels and volume confirmation.

name = "6h_Weekly_Pivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Calculate R3 and S3 levels
    # R3 = High + 2*(Pivot - Low)
    # S3 = Low - 2*(High - Pivot)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend conditions
        uptrend = close[i] > ema50_daily_aligned[i]
        downtrend = close[i] < ema50_daily_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above weekly R3 + uptrend + volume spike
            if close[i] > weekly_r3_aligned[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + downtrend + volume spike
            elif close[i] < weekly_s3_aligned[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below weekly pivot OR trend reversal
            if close[i] < weekly_pivot[-1] if len(weekly_pivot) > 0 else False or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly pivot OR trend reversal
            if close[i] > weekly_pivot[-1] if len(weekly_pivot) > 0 else False or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals