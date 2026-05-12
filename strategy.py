#!/usr/bin/env python3
"""
6h_Aroon_DMI_Trend_Filter_VolumeSpike
Hypothesis: Aroon oscillator (trend strength) combined with DMI (+DI/-DI crossover) and volume spike (1.5x 20-period average) captures high-probability trending moves on 6b timeframe. Uses 1d ADX as regime filter to avoid ranging markets. Works in bull/bear by following 1d trend direction via ADX > 25.
"""

name = "6h_Aroon_DMI_Trend_Filter_VolumeSpike"
timeframe = "6h"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # 1d ADX for regime filter (trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate +DM, -DM, TR
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))

    # Prepend first values
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])

    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result

    period = 14
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)

    # Shift ADX to avoid look-ahead (use previous day's value)
    adx_prev = np.roll(adx, 1)
    adx_prev[0] = np.nan
    adx_1d = adx_prev

    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # Aroon oscillator (6b timeframe)
    def aroon(high, low, period=25):
        n = len(high)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)
        for i in range(period, n):
            # Find highest high in lookback period
            high_idx = np.argmax(high[i-period:i+1]) + i - period
            # Find lowest low in lookback period
            low_idx = np.argmin(low[i-period:i+1]) + i - period
            aroon_up[i] = ((period - (i - high_idx)) / period) * 100
            aroon_down[i] = ((period - (i - low_idx)) / period) * 100
        return aroon_up - aroon_down  # Oscillator: -100 to 100

    aroon_osc = aroon(high, low, 25)

    # DMI crossover signal (6b timeframe)
    # Calculate +DM, -DM, TR for 6b
    plus_dm_6h = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm_6h = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
    tr_6h = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))

    plus_dm_6h = np.concatenate([[0], plus_dm_6h])
    minus_dm_6h = np.concatenate([[0], minus_dm_6h])
    tr_6h = np.concatenate([[high[0] - low[0]], tr_6h])

    atr_6h = wilder_smooth(tr_6h, 14)
    plus_di_6h = 100 * wilder_smooth(plus_dm_6h, 14) / atr_6h
    minus_di_6h = 100 * wilder_smooth(minus_dm_6h, 14) / atr_6h

    # DMI crossover: +DI crosses above/below -DI
    dmi_cross_up = np.zeros(n, dtype=bool)
    dmi_cross_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if (plus_di_6h[i] > minus_di_6h[i] and 
            plus_di_6h[i-1] <= minus_di_6h[i-1]):
            dmi_cross_up[i] = True
        if (plus_di_6h[i] < minus_di_6h[i] and 
            plus_di_6h[i-1] >= minus_di_6h[i-1]):
            dmi_cross_down[i] = True

    # Volume spike: >1.5x 20-period average (6b)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(aroon_osc[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Aroon > 50 (uptrend) + DMI crossover up + ADX > 25 + volume spike
            if (aroon_osc[i] > 50 and 
                dmi_cross_up[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Aroon < -50 (downtrend) + DMI crossover down + ADX > 25 + volume spike
            elif (aroon_osc[i] < -50 and 
                  dmi_cross_down[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Aroon < 0 (trend weakening) or ADX < 20 (ranging)
            if aroon_osc[i] < 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Aroon > 0 (trend weakening) or ADX < 20 (ranging)
            if aroon_osc[i] > 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals