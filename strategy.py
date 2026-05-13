#!/usr/bin/env python3
# 12h_1w_Volume_Price_Action_Strategy
# Hypothesis: Use 1w price action (weekly high/low) as primary support/resistance, combined with 12h volume confirmation and trend alignment.
# Long when price breaks above weekly high with 12h uptrend and volume spike.
# Short when price breaks below weekly low with 12h downtrend and volume spike.
# Weekly levels provide strong institutional support/resistance, effective in both trending and ranging markets.
# Volume confirmation reduces false breakouts, while trend filter ensures alignment with higher timeframe momentum.
# Target: 12-30 trades/year per symbol to stay within optimal range for 12h timeframe.

name = "12h_1w_Volume_Price_Action_Strategy"
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

    # Get 1w data for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high and low from previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # 12h trend: EMA34
    close_12h = df_1w['close'].values  # Using 1w close for trend (more stable)
    ema34_1w = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly high + 12h uptrend + volume spike
            if close[i] > weekly_high_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly low + 12h downtrend + volume spike
            elif close[i] < weekly_low_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly low or trend reversal
            if close[i] < weekly_low_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly high or trend reversal
            if close[i] > weekly_high_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals