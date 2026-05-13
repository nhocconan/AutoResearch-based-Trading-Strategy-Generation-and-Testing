#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume
# Hypothesis: Use hourly timeframe for entry timing with 4h trend filter and 1d volume confirmation.
# Long when price breaks above 4h R3 in 4h uptrend with 1d volume spike, short when price breaks below 4h S3 in 4h downtrend with 1d volume spike.
# Exit when price returns to 4h pivot point or 4h trend changes.
# Uses 4h for signal direction (reducing trade frequency) and 1h only for precise entry timing.
# Session filter (08-20 UTC) to avoid low-volume Asian session noise.
# Designed for 15-37 trades/year target on 1h timeframe.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
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

    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 4h data for Camarilla pivot calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Camarilla pivot levels: R3, S3, PP
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    
    r3_4h = close_4h + ((high_4h - low_4h) * 1.2500)
    s3_4h = close_4h - ((high_4h - low_4h) * 1.2500)
    pp_4h = (high_4h + low_4h + close_4h) / 3
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h.values)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h.values)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h.values)

    # Get 4h data for EMA trend filter (20-period)
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)

    # Get 1d data for volume filter: >1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_avg_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if not in trading session or any required value is NaN
        if not in_session[i] or \
           (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pp_4h_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 4h R3 + price above 4h EMA20 (uptrend) + 1d volume spike
            if (close[i] > r3_4h_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h S3 + price below 4h EMA20 (downtrend) + 1d volume spike
            elif (close[i] < s3_4h_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to 4h pivot point (PP) or trend changes (price below 4h EMA20)
            if (close[i] <= pp_4h_aligned[i] or close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to 4h pivot point (PP) or trend changes (price above 4h EMA20)
            if (close[i] >= pp_4h_aligned[i] or close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals