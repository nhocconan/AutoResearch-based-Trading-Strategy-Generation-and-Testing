#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Camarilla pivot levels (R1/S1) on 12h with daily trend filter and volume confirmation
# capture significant trend moves while avoiding whipsaw. Daily trend ensures alignment with
# higher-timeframe momentum, reducing false signals. Volume confirms breakout strength.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA20 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla pivot levels for 12h (based on previous day's OHLC)
    # We'll use the previous day's close, high, low to calculate today's Camarilla levels
    # Since we're on 12h timeframe, we need to align the daily pivot levels to each 12h bar
    
    # Get daily OHLC for pivot calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
        
    # Calculate Camarilla levels from previous day's data
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # R2 = close + ((high - low) * 1.166)
    # R1 = close + ((high - low) * 1.083)
    # S1 = close - ((high - low) * 1.083)
    # S2 = close - ((high - low) * 1.166)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Calculate R1 and S1 levels
    r1 = prev_close + ((prev_high - prev_low) * 1.083)
    s1 = prev_close - ((prev_high - prev_low) * 1.083)
    
    # Align daily pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s1)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and daily uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and daily downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below S1 or daily trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above R1 or daily trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals