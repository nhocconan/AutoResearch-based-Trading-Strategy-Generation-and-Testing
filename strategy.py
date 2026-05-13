#!/usr/bin/env python3
# 4h_Donchian_Breakout_12hTrend_Volume
# Hypothesis: Breakout above/below 4h Donchian channel (20) in the direction of 12h trend (EMA50), confirmed by volume spike (volume > 2.0 * 24-period average). Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend). Low frequency due to strict entry conditions and volume confirmation.

name = "4h_Donchian_Breakout_12hTrend_Volume"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 4h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    # 4h Donchian channel (20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > 4h Donchian upper + 12h uptrend + volume spike
            if close[i] > high_20[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < 4h Donchian lower + 12h downtrend + volume spike
            elif close[i] < low_20[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 4h Donchian lower OR trend reversal
            if close[i] < low_20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above 4h Donchian upper OR trend reversal
            if close[i] > high_20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals