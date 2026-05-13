#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Price reacts to Camarilla pivot levels (R1/S1) derived from 4h timeframe. 
# Go long when price breaks above R1 with 4h uptrend and volume confirmation on 1h.
# Go short when price breaks below S1 with 4h downtrend and volume confirmation on 1h.
# Uses 4h for signal direction (trend and pivot levels), 1h for entry timing.
# Session filter (08-20 UTC) reduces noise trades. Position size 0.20 to control drawdown.
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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

    # Get 4h data for Camarilla pivot calculation and trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 4h bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    camarilla_width = (high_4h - low_4h) * 1.1 / 12
    r1 = close_4h + camarilla_width
    s1 = close_4h - camarilla_width
    
    # 4h trend: EMA21
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume spike: volume > 2.0 * 6-period average (6 hours worth at 1h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema21_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if session_filter[i]:
            if position == 0:
                # LONG: Close > R1 + 4h uptrend + volume spike
                if close[i] > r1_aligned[i] and close[i] > ema21_4h_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                # SHORT: Close < S1 + 4h downtrend + volume spike
                elif close[i] < s1_aligned[i] and close[i] < ema21_4h_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Close below S1 or trend reversal
                if close[i] < s1_aligned[i] or close[i] < ema21_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # EXIT SHORT: Close above R1 or trend reversal
                if close[i] > r1_aligned[i] or close[i] > ema21_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals