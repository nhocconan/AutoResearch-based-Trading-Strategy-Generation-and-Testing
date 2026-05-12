#160133: 4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Price breaking above/below Camarilla R3/S3 levels with 12-hour EMA50 trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Works in bull/bear by following the higher timeframe trend direction. Uses 4h timeframe with 12h EMA50 trend filter for higher timeframe context.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate 12-hour high, low, close for Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = high_12h - low_12h
    r3_level = close_12h + 1.1 * camarilla_range / 2
    s3_level = close_12h - 1.1 * camarilla_range / 2

    # Align Camarilla levels to 4h timeframe
    r3_level_aligned = align_htf_to_ltf(prices, df_12h, r3_level)
    s3_level_aligned = align_htf_to_ltf(prices, df_12h, s3_level)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after EMA50 warmup
        if (np.isnan(r3_level_aligned[i]) or np.isnan(s3_level_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + EMA50 uptrend + volume confirmation
            if (close[i] > r3_level_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + EMA50 downtrend + volume confirmation
            elif (close[i] < s3_level_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals