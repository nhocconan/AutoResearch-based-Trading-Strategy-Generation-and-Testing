# [12h_Camarilla_R1_S1_Breakout_1wTrend_Volume] - 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Hypothesis: Price breaking above/below Camarilla R1/S1 levels on 12h timeframe, filtered by 1w trend (price above/below 200 EMA),
# and confirmed by volume spikes. Works in bull markets via long entries in uptrends, and in bear markets via short entries in downtrends.
# Volume spike ensures institutional participation. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1w EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Calculate 12h Camarilla pivot levels from previous day
    # Camarilla uses previous day's OHLC
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    range_hl = prev_day_high - prev_day_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Volume spike: current > 2.0x average of last 24 bars (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after EMA200 warmup
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1w uptrend + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1w downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend breaks
            if close[i] < s1_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend breaks
            if close[i] > r1_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals