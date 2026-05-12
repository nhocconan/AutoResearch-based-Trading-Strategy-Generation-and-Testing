# 160088: 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Price breaking above/below weekly Camarilla R3/S3 levels with 1-week EMA34 trend filter and volume confirmation captures strong trending moves in both bull and bear markets. Uses 12h timeframe with 1-week EMA34 trend filter for higher timeframe context, reducing whipsaw and improving trend following. Weekly timeframe provides robust trend filter less prone to noise than daily.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1-week high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = high_1w - low_1w
    r3_level = close_1w + 1.1 * camarilla_range / 2
    s3_level = close_1w - 1.1 * camarilla_range / 2

    # Align Camarilla levels to 12h timeframe
    r3_level_aligned = align_htf_to_ltf(prices, df_1w, r3_level)
    s3_level_aligned = align_htf_to_ltf(prices, df_1w, s3_level)

    # 1-week EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after EMA34 warmup
        if (np.isnan(r3_level_aligned[i]) or np.isnan(s3_level_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + EMA34 uptrend + volume confirmation
            if (close[i] > r3_level_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 + EMA34 downtrend + volume confirmation
            elif (close[i] < s3_level_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA34 (trend reversal)
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above EMA34 (trend reversal)
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals