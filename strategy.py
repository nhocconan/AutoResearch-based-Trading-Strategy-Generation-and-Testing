#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Filter
# Hypothesis: Long when price breaks above Camarilla R1 level with 12h uptrend and volume spike; short when price breaks below S1 level with 12h downtrend and volume spike.
# Exit on opposite Camarilla level breach or trend reversal. Uses 12h trend filter to avoid counter-trend whipsaws. Designed to work in both bull (trend-following breakouts) and bear (counter-trend bounces) markets by following the higher timeframe trend.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Filter"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate daily high, low, close for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_S1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 in 12h uptrend with volume spike
            if close[i] > camarilla_R1_aligned[i]:
                if close[i] > ema_50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below Camarilla S1 in 12h downtrend with volume spike
            elif close[i] < camarilla_S1_aligned[i]:
                if close[i] < ema_50_12h_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or trend turns down
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_50_12h_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or trend turns up
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_50_12h_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals