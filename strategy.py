#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Uses weekly Donchian channels to define trend direction, enters on breakout of daily Donchian
# channels in the direction of the weekly trend. Volume filter ensures breakout has participation.
# Designed for low trade frequency (~10-20/year) to minimize fee drift. Trend filter avoids
# counter-trend trades in bear markets like 2022 and 2025.

name = "1d_DonchianBreakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period) for trend
    high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (high_1w + low_1w) / 2
    donchian_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Calculate daily Donchian channels (20-period) for entry
    high_d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper_d = high_d
    donchian_lower_d = low_d
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(donchian_mid_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above daily Donchian upper with weekly uptrend and volume
            if (close[i] > donchian_upper_d[i] and 
                close[i] > donchian_mid_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below daily Donchian lower with weekly downtrend and volume
            elif (close[i] < donchian_lower_d[i] and 
                  close[i] < donchian_mid_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily Donchian lower (reversal signal)
            if close[i] < donchian_lower_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily Donchian upper (reversal signal)
            if close[i] > donchian_upper_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals