#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Price breaking above/below 4-hour Camarilla R1/S1 levels with 1-day volume confirmation and 4-hour trend filter captures strong trending moves in 1h timeframe while avoiding false breakouts. Works in bull/bear by following the 4h trend direction. Uses 1h timeframe with 4h trend and 1d volume filters to reduce noise and false signals.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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

    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h high, low, close for Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate 4h Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range_4h = high_4h - low_4h
    r1_level_4h = close_4h + 1.1 * camarilla_range_4h / 12
    s1_level_4h = close_4h - 1.1 * camarilla_range_4h / 12

    # Align Camarilla levels to 1h timeframe
    r1_level_aligned = align_htf_to_ltf(prices, df_4h, r1_level_4h)
    s1_level_aligned = align_htf_to_ltf(prices, df_4h, s1_level_4h)

    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume confirmation: >1.8x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = vol_1d > (1.8 * vol_ma_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after EMA20 warmup
        if (np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_confirm_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 4h EMA20 uptrend + 1d volume confirmation
            if (close[i] > r1_level_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                volume_confirm_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h EMA20 downtrend + 1d volume confirmation
            elif (close[i] < s1_level_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  volume_confirm_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h EMA20 (trend reversal)
            if close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h EMA20 (trend reversal)
            if close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals