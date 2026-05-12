# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with 1-day trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Works in bull/bear by following the daily trend direction. Uses 12h timeframe with daily trend filter for higher timeframe context.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate daily high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * camarilla_range / 12
    s1_level = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to 12h timeframe
    r1_level_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_level_aligned = align_htf_to_ltf(prices, df_1d, s1_level)

    # 1d trend filter: close > close[1] for uptrend, close < close[1] for downtrend
    trend_up = close_1d > np.roll(close_1d, 1)
    trend_down = close_1d < np.roll(close_1d, 1)
    # Handle first element
    trend_up[0] = False
    trend_down[0] = False
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        if (np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volume confirmation
            if (close[i] > r1_level_aligned[i] and 
                trend_up_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volume confirmation
            elif (close[i] < s1_level_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversion to mean)
            if close[i] < s1_level_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversion to mean)
            if close[i] > r1_level_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals