#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_WeeklyTrend
# Hypothesis: On daily chart, buy when price breaks above Camarilla R1 level with volume confirmation and weekly uptrend,
# sell when price breaks below Camarilla S1 level with volume confirmation and weekly downtrend.
# Camarilla levels provide high-probability reversal points; breakouts indicate institutional participation.
# Weekly trend filter avoids counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend.
timeframe = "1d"
name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA trend filter (8-period)
    ema_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Previous day's Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R1 and S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Volume spike: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + weekly uptrend + volume spike
            if close[i] > R1[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + weekly downtrend + volume spike
            elif close[i] < S1[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 or weekly trend turns down
            if close[i] < S1[i] or ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 or weekly trend turns up
            if close[i] > R1[i] or ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals