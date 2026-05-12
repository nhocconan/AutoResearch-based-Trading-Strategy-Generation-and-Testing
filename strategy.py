# 160064: 1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeS
# Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with weekly EMA34 trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Uses daily timeframe with weekly EMA34 trend filter for higher timeframe context. Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets by following weekly trend direction.

#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeS"
timeframe = "1d"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly high, low, close for EMA34 trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate 1-week EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Calculate daily high, low, close for Camarilla levels
    high_1d = high
    low_1d = low
    close_1d = close

    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * camarilla_range / 12
    s1_level = close_1d - 1.1 * camarilla_range / 12

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after EMA34 warmup
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_level[i]) or 
            np.isnan(s1_level[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + weekly EMA34 uptrend + volume confirmation
            if (close[i] > r1_level[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + weekly EMA34 downtrend + volume confirmation
            elif (close[i] < s1_level[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA34 (trend reversal)
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA34 (trend reversal)
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals