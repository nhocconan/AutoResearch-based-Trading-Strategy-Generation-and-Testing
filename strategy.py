#!/usr/bin/env python3
# 1D_Camarilla_R1_S1_WeeklyTrend_Volume
# Hypothesis: Daily Camarilla R1/S1 level breakouts filtered by weekly trend and volume spikes.
# Uses Camarilla pivot levels from daily data as support/resistance.
# Trend filter: weekly EMA50 (only trade in direction of higher timeframe trend).
# Volume confirmation: current volume > 2.0 x 20-period average.
# Designed to work in both bull and bear markets by following weekly trend direction.
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "1D_Camarilla_R1_S1_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for HTF trend filter
    df_weekly = get_htf_data(prices, '1w')

    # Calculate Camarilla levels from daily data
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.6*(high-low), R1 = close + 0.382*(high-low)
    #          S1 = close - 0.382*(high-low), S2 = close - 0.6*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r1 = close + 0.382 * (high - low)
    camarilla_s1 = close - 0.382 * (high - low)

    # Trend filter: weekly EMA50
    ema50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Camarilla R1 in uptrend with volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema50_weekly_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 in downtrend with volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema50_weekly_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals