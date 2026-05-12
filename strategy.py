#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: On 12h timeframe, trade Camarilla pivot level (R3/S3) breakouts with weekly EMA trend filter and volume confirmation.
# Long when: price breaks above R3, volume spike, and price above weekly EMA50
# Short when: price breaks below S3, volume spike, and price below weekly EMA50
# Uses weekly trend to avoid counter-trend trades and volume to avoid false breakouts.
# Targets 15-30 trades/year by requiring confluence of breakout, volume, and trend.
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extreme levels).

name = "12h_Camarilla_Pivot_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (EMA50) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate EMA50 on weekly close
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Get daily data for Camarilla pivot calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day's OHLC
    # Use previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN since there's no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla calculations
    range_1d = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + range_1d * 1.1 / 2
    s3 = close_1d_prev - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume confirmation: current volume > 2.0x average of last 24 periods (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: price breaks above R3, volume spike, and above weekly EMA50
            if close[i] > r3_aligned[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3, volume spike, and below weekly EMA50
            elif close[i] < s3_aligned[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below S3 or closes below weekly EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R3 or closes above weekly EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals