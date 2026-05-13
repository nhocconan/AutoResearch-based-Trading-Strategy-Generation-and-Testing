#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout of daily Camarilla R1/S1 levels on 12h timeframe with daily trend filter and volume confirmation.
# Long when price breaks above R1 during daily uptrend with volume spike.
# Short when price breaks below S1 during daily downtrend with volume spike.
# Exit on opposite Camarilla level breach or trend reversal.
# Designed for low-frequency trading (12-37 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels: R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    # where C, H, L are daily close, high, low
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    camarilla_r1 = daily_close + 1.1 * (daily_high - daily_low) / 12
    camarilla_s1 = daily_close - 1.1 * (daily_high - daily_low) / 12
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: >1.5x 20-period average on 12h chart
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 during daily uptrend with volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i-1] <= camarilla_r1_aligned[i-1]:
                if close[i] > ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 during daily downtrend with volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i-1] >= camarilla_s1_aligned[i-1]:
                if close[i] < ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] and close[i-1] >= camarilla_s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] and close[i-1] <= camarilla_r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals