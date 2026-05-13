#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Camarilla pivot breakouts on 1h with 4h EMA trend filter and 1d volume spike capture institutional breakout moves while avoiding chop. Works in bull/bear via 4h trend filter and volume confirmation. Targets 15-30 trades/year via tight entry conditions.

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

    # Calculate Camarilla levels for 1h (using previous bar's range)
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    # We use previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_hl = prev_high - prev_low
    camarilla_r1 = prev_close + 1.1 * range_hl / 12
    camarilla_s1 = prev_close - 1.1 * range_hl / 12

    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # 1d volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 4h EMA50 uptrend + 1d volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema50_4h_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Close below S1 + 4h EMA50 downtrend + 1d volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend change
            if close[i] < camarilla_s1[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend change
            if close[i] > camarilla_r1[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals