#!/usr/bin/env python3
# 12h_1d_Camarilla_R3S3_Breakout_Trend
# Hypothesis: Use Camarilla R3/S3 levels from daily candles as breakout levels with daily trend filter and volume confirmation.
# Long when price breaks above R3 with daily uptrend and volume spike.
# Short when price breaks below S3 with daily downtrend and volume spike.
# Exit on opposite Camarilla level or trend reversal.
# Designed to capture trends with controlled frequency to avoid fee drag.

name = "12h_1d_Camarilla_R3S3_Breakout_Trend"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    range_1d = high_1d_prev - low_1d_prev
    R3 = close_1d_prev + 1.125 * range_1d
    S3 = close_1d_prev - 1.125 * range_1d
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    # Align 1d indicators to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above R3 + uptrend + volume spike
            if price_above_R3 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + downtrend + volume spike
            elif price_below_S3 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend reversal
            if price_below_S3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend reversal
            if price_above_R3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals