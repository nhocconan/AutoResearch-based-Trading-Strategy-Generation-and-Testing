#!/usr/bin/env python3
# 1d_Donchian_20_1wTrend_Volume_Spike
# Hypothesis: 1d Donchian(20) breakout with weekly EMA34 trend filter and volume spike (2.0x 20-day avg). Designed for low trade frequency (<=25/year) to avoid fee drag. Works in bull (breakouts with momentum) and bear (mean reversion via trend filter) by requiring trend alignment, reducing false breakouts.

name = "1d_Donchian_20_1wTrend_Volume_Spike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # LONG: Price breaks above Donchian upper band with volume spike in weekly uptrend
        if position == 0 and roll_max[i] > 0 and not np.isnan(roll_max[i]) and \
           high[i] > roll_max[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
            signals[i] = 0.25
            position = 1
        # SHORT: Price breaks below Donchian lower band with volume spike in weekly downtrend
        elif position == 0 and roll_min[i] > 0 and not np.isnan(roll_min[i]) and \
             low[i] < roll_min[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
            signals[i] = -0.25
            position = -1
        # EXIT LONG: Price falls below Donchian lower band
        elif position == 1 and roll_min[i] > 0 and not np.isnan(roll_min[i]) and \
             low[i] < roll_min[i]:
            signals[i] = 0.0
            position = 0
        # EXIT SHORT: Price rises above Donchian upper band
        elif position == -1 and roll_max[i] > 0 and not np.isnan(roll_max[i]) and \
             high[i] > roll_max[i]:
            signals[i] = 0.0
            position = 0
        else:
            signals[i] = 0.0
    
    return signals