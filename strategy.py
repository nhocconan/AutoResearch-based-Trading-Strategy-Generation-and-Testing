#!/usr/bin/env python3
# 1h_4h_Trend_Filtered_Breakout_With_Volume
# Hypothesis: Use 4h trend as primary direction filter, enter long/short on 1h breakouts above/below 4h Donchian channels with volume confirmation. Exit on trend reversal or opposite breakout.
# Works in bull: 4h uptrend + 1h breakout above channel = long. Works in bear: 4h downtrend + 1h breakdown below channel = short.
# Volume confirmation filters weak breakouts. Trend filter reduces whipsaw in sideways markets.
# Target frequency: 15-30 trades/year by requiring 4h trend alignment + volume spike + breakout.

name = "1h_4h_Trend_Filtered_Breakout_With_Volume"
timeframe = "1h"
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

    # Get 4h data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume spike: volume > 2.0 * 12-period average (6h worth at 1h)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > 2.0 * vol_ma_12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend + price breaks above 4h Donchian high + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + price breaks below 4h Donchian low + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h downtrend OR price breaks below 4h Donchian low
            if close[i] < ema50_4h_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h uptrend OR price breaks above 4h Donchian high
            if close[i] > ema50_4h_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals