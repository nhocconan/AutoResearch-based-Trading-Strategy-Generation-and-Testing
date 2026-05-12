# 1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Price breaking above/below weekly Camarilla R3/S3 levels with 1-week trend filter and volume confirmation
# captures strong multi-week moves while avoiding false breakouts. Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike (2x average) confirms institutional interest. Works in bull/bear by following 1-week trend direction.
# Target: 15-25 trades/year per symbol (<100 total over 4 years) to minimize fee drag.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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

    # Calculate weekly Camarilla levels (R3/S3)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Shift by 1 to use previous week's data
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan

    # Weekly Camarilla R3 and S3
    camarilla_upper_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 4
    camarilla_lower_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 4

    # Align weekly Camarilla levels to daily timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1w, camarilla_upper_1w)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1w, camarilla_lower_1w)

    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume spike: >2.0x 20-day average (daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Camarilla R3 + weekly uptrend + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Camarilla S3 + weekly downtrend + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly Camarilla S3 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly Camarilla R3 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals