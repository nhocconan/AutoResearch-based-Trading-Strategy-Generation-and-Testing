# 1D_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA trend filter and volume spikes.
# Uses daily pivot levels for mean-reversion breakouts and weekly EMA for trend filter.
# Works in bull via breakouts and in bear via range-bound reversals.
# Target: 15-25 trades/year per symbol.

name = "1D_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate daily Camarilla pivot levels
    # Based on previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # First day has no previous data
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_r1 = close + (range_hl * 1.1 / 12)
    camarilla_s1 = close - (range_hl * 1.1 / 12)
    # Note: These are today's levels based on yesterday's range

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R1 + weekly uptrend + volume spike
            if close[i] > camarilla_r1[i] and close[i] > ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1 + weekly downtrend + volume spike
            elif close[i] < camarilla_s1[i] and close[i] < ema20_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla pivot or weekly trend turns down
            if close[i] < camarilla_r1[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla S1 or weekly trend turns up
            if close[i] > camarilla_s1[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals