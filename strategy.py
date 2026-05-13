#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA trend filter and volume spike.
# Uses 4h timeframe for signal direction and 1h for entry timing to reduce noise.
# Designed for 15-30 trades/year on 1h timeframe to minimize fee drag.
# Works in bull/bear markets: long when price breaks above R1 with volume and above 4h EMA;
# short when breaks below S1 with volume and below 4h EMA.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')

    # Calculate Camarilla pivot levels from previous 4h bar
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate R1 and S1 using standard formula
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 1h timeframe (values available after 4h bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)

    # 4h EMA34 trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after sufficient warmup for EMA34
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and above 4h EMA34
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume spike and below 4h EMA34
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or closes below 4h EMA34
            if close[i] < s1_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or closes above 4h EMA34
            if close[i] > r1_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals