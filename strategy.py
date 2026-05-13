#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade breakouts of Camarilla R1/S1 levels on 4h timeframe, filtered by 1-day trend and volume spike.
# Long when price breaks above R1 during 1-day uptrend with volume > 1.5x 20-bar average.
# Short when price breaks below S1 during 1-day downtrend with volume > 1.5x 20-bar average.
# Exit on break of opposite level (S1 for long, R1 for short) or trend reversal.
# Uses daily trend filter to avoid counter-trend whipsaws, targeting 20-40 trades/year per symbol.
# Designed to work in bull markets (trend continuation) and bear markets (mean reversion at extremes).

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift to get previous day's values
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous
    
    # Calculate Camarilla levels
    camarilla_width = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 in 1-day uptrend with volume spike
            if close[i] > r1_aligned[i]:
                if close[i] > ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 in 1-day downtrend with volume spike
            elif close[i] < s1_aligned[i]:
                if close[i] < ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend turns down
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend turns up
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals