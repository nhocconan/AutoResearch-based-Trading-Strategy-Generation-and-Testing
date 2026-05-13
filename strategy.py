#!/usr/bin/env python3
# 1D_1W_Camarilla_Pivot_Support_Resistance_Bounce
# Hypothesis: On daily timeframe, price tends to respect weekly Camarilla pivot levels (S3/S4 and R3/R4) as major support/resistance.
# Go long when price touches weekly S3/S4 with bullish rejection (close > open) and volume confirmation.
# Go short when price touches weekly R3/R4 with bearish rejection (close < open) and volume confirmation.
# Uses weekly timeframe for structural levels (more reliable in ranging/ bear markets) and daily for execution.
# Volume spike filters for institutional interest. Works in both bull/bear by fading extremes at proven weekly levels.

name = "1D_1W_Camarilla_Pivot_Support_Resistance_Bounce"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels for weekly timeframe
    # Formula: R4 = close + (high - low) * 1.1/2, R3 = close + (high - low) * 1.1/4, etc.
    # We focus on S3, S4, R3, R4 as extreme levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels
    diff = weekly_high - weekly_low
    S4 = weekly_close - diff * 1.1 / 2
    S3 = weekly_close - diff * 1.1 / 4
    R3 = weekly_close + diff * 1.1 / 4
    R4 = weekly_close + diff * 1.1 / 2
    
    # Align weekly Camarilla levels to daily timeframe
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    # Bullish/bearish rejection candles
    bullish_rejection = close > open_price  # close above open
    bearish_rejection = close < open_price  # close below open
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(S4_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(R4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches weekly S3/S4 with bullish rejection and volume spike
            if ((low[i] <= S3_aligned[i] or low[i] <= S4_aligned[i]) and 
                bullish_rejection[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches weekly R3/R4 with bearish rejection and volume spike
            elif ((high[i] >= R3_aligned[i] or high[i] >= R4_aligned[i]) and 
                  bearish_rejection[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back above weekly S3 or shows weakness
            if close[i] > S3_aligned[i] or not bullish_rejection[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back below weekly R3 or shows weakness
            if close[i] < R3_aligned[i] or not bearish_rejection[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals