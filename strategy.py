#!/usr/bin/env python3
# 4h_PriceChannel_Breakout_DMIFilter
# Hypothesis: Use 4h Donchian(20) breakout with DMI trend filter (ADX>25) and volume confirmation (vol>1.5x20avg) to capture strong trends.
# Enter long on upper band breakout, short on lower band breakout. Exit when price crosses the middle (20-period average of high/low).
# Works in bull (trend continuation) and bear (trend reversals) by filtering with DMI to avoid whipsaws.
# Target: 25-40 trades/year on 4h to minimize fee drag.

name = "4h_PriceChannel_Breakout_DMIFilter"
timeframe = "4h"
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

    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max_20 + low_min_20) / 2.0

    # Calculate DMI (ADX) components for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (14-period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = np.where(tr_sum > 0, 100 * plus_dm_sum / tr_sum, 0)
    minus_di = np.where(tr_sum > 0, 100 * minus_dm_sum / tr_sum, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above upper Donchian + ADX>25 + volume spike
            if (close[i] > high_max_20[i] and 
                adx[i] > 25 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower Donchian + ADX>25 + volume spike
            elif (close[i] < low_min_20[i] and 
                  adx[i] > 25 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below middle of channel
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above middle of channel
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals