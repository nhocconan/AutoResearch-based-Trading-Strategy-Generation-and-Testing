#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakout with volume confirmation and 1d EMA50 trend filter on 4h timeframe.
# Uses Camarilla pivot levels calculated from previous day's high/low/close. Long when price breaks above R1 with volume spike and price above 1d EMA50.
# Short when price breaks below S1 with volume spike and price below 1d EMA50.
# Exit when price crosses back through the Camarilla pivot point (central level).
# Designed for 25-40 trades/year to minimize fee drift. Works in both bull and bear by capturing institutional breakout levels with trend alignment.

name = "4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume"
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

    # Camarilla pivot levels (based on previous day's range)
    # R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, R2 = close + (high-low)*1.1/6, R1 = close + (high-low)*1.1/12
    # S1 = close - (high-low)*1.1/12, S2 = close - (high-low)*1.1/6, S3 = close - (high-low)*1.1/4, S4 = close - (high-low)*1.1/2
    # Pivot = (high + low + close)/3
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_P = np.full(n, np.nan)  # pivot point for exit

    for i in range(1, n):
        # Use previous day's data (we'll get this from 1d data via HTF)
        pass  # Will fill in HTF section

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous day's OHLC
    camarilla_R1_1d = np.full(len(close_1d), np.nan)
    camarilla_S1_1d = np.full(len(close_1d), np.nan)
    camarilla_P_1d = np.full(len(close_1d), np.nan)

    for i in range(1, len(close_1d)):
        # Previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_P_1d[i] = (prev_high + prev_low + prev_close) / 3
        camarilla_R1_1d[i] = prev_close + (range_val * 1.1 / 12)
        camarilla_S1_1d[i] = prev_close - (range_val * 1.1 / 12)

    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P_1d)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above 1d EMA50 (uptrend)
            if close[i] > camarilla_R1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below 1d EMA50 (downtrend)
            elif close[i] < camarilla_S1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point (mean reversion to mean)
            if close[i] < camarilla_P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point
            if close[i] > camarilla_P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals