#!/usr/bin/env python3
"""
4h_Ichimoku_Kijun_Bounce_1dTrend_Volume
Hypothesis: Enter long when price bounces above Ichimoku Kijun-sen (base line) with price above Kumo (cloud) and Tenkan above Kijun, aligned with 1d EMA50 uptrend and volume confirmation. Enter short when price breaks below Kijun with price below Kumo and Tenkan below Kijun, aligned with 1d EMA50 downtrend and volume confirmation. Ichimoku provides dynamic support/resistance and trend context, reducing whipsaw in ranging markets. Target: 20-50 trades/year on 4h timeframe with controlled risk via trend and volume filters.
"""
name = "4h_Ichimoku_Kijun_Bounce_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for 1d EMA50 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA50 trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Ichimoku components on 4h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2

    # Volume spike: current > 2.0x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Senkou B warmup
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Kumo (cloud) boundaries: Senkou A and Senkou B
        upper_kumo = np.maximum(senkou_a[i], senkou_b[i])
        lower_kumo = np.minimum(senkou_a[i], senkou_b[i])

        if position == 0:
            # LONG: price > Kijun, price > Kumo (bullish), Tenkan > Kijun, aligned with 1d uptrend, volume spike
            if (close[i] > kijun[i] and 
                close[i] > upper_kumo and 
                tenkan[i] > kijun[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < Kijun, price < Kumo (bearish), Tenkan < Kijun, aligned with 1d downtrend, volume spike
            elif (close[i] < kijun[i] and 
                  close[i] < lower_kumo and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Kijun or price < Kumo or trend breaks
            if (close[i] < kijun[i] or 
                close[i] < lower_kumo or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Kijun or price > Kumo or trend breaks
            if (close[i] > kijun[i] or 
                close[i] > upper_kumo or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals