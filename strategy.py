#!/usr/bin/env python3
"""
1h_Advanced_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_Spike
Hypothesis: Combines 4h trend filter (EMA50) with 1d volume confirmation and 1h price action breaking 1d Camarilla R1/S1 levels.
Uses 4h EMA50 for trend direction and 1d volume spike (1.5x 20-day average) to confirm institutional participation.
Trades only during 08:00-20:00 UTC to avoid low-liquidity periods.
Designed for 1h timeframe with tight entry conditions to limit trades to 15-30/year.
Works in bull/bear markets by following 4h trend direction and requiring volume confirmation.
"""

name = "1h_Advanced_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_Spike"
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

    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data ONCE before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')

    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Calculate 1d Camarilla levels from previous day's data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Shift by 1 to use previous day's data
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan

    camarilla_upper = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_lower = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4

    # Align Camarilla levels to 1h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)

    # Calculate 1d volume spike: >1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)

    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Skip if any required data is NaN
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 4h EMA50 uptrend + 1d volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 4h EMA50 downtrend + 1d volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals