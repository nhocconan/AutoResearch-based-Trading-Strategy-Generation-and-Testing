# 4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Breakout above daily Camarilla R1 or below S1 in the direction of 1d EMA50 trend, confirmed by volume spike (>2x 4-period average).
# Uses daily pivot levels for structure, trend filter for direction, volume for confirmation.
# Designed to work in both bull and bear markets by following the daily trend.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot point
    pivot = (daily_high + daily_low + daily_close) / 3
    # Camarilla levels
    R1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    S1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (equivalent to 8 hours)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > daily R1 + daily uptrend + volume spike
            if close[i] > R1_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < daily S1 + daily downtrend + volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below daily pivot OR trend reversal
            if close[i] < pivot_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above daily pivot OR trend reversal
            if close[i] > pivot_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals