# 160074: 1h_Camarilla_R3_S3_Breakout_4hEMA12_Trend_VolumeSession
# Hypothesis: Price breaking above/below Camarilla R3/S3 levels with 4-hour EMA12 trend filter, volume confirmation, and session filter (08-20 UTC) captures strong trending moves while reducing noise. Uses 1h timeframe with 4h EMA12 trend filter for higher timeframe context. Session filter avoids low-liquidity periods. Target: 15-37 trades/year (60-150 over 4 years) by using tight entry conditions.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA12_Trend_VolumeSession"
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

    # Calculate 4-hour high, low, close for Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = high_4h - low_4h
    r3_level = close_4h + 1.1 * camarilla_range / 2
    s3_level = close_4h - 1.1 * camarilla_range / 2

    # Align Camarilla levels to 1h timeframe
    r3_level_aligned = align_htf_to_ltf(prices, df_4h, r3_level)
    s3_level_aligned = align_htf_to_ltf(prices, df_4h, s3_level)

    # 4h EMA12 trend filter
    ema_12_4h = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_12_4h)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after EMA12 warmup and session filter
        if (np.isnan(r3_level_aligned[i]) or np.isnan(s3_level_aligned[i]) or 
            np.isnan(ema_12_4h_aligned[i]) or np.isnan(volume_confirm[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + EMA12 uptrend + volume confirmation + session
            if (close[i] > r3_level_aligned[i] and 
                close[i] > ema_12_4h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 + EMA12 downtrend + volume confirmation + session
            elif (close[i] < s3_level_aligned[i] and 
                  close[i] < ema_12_4h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA12 (trend reversal)
            if close[i] < ema_12_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above EMA12 (trend reversal)
            if close[i] > ema_12_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals