#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Uses weekly Donchian channels to establish primary trend, enters on daily breakouts
# in the direction of the weekly trend. Volume filter ensures breakout has participation.
# Designed for low trade frequency (~10-25/year) to minimize fee drift in bear markets.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion
# at weekly support/resistance when combined with volatility filter.

name = "1d_DonchianBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period) for trend
    high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_1w = align_htf_to_ltf(prices, df_1w, high_1w)
    donchian_low_1w = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Daily Donchian breakout levels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: current volume > 20-day average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above daily Donchian high with weekly uptrend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > donchian_high_1w[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below daily Donchian low with weekly downtrend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < donchian_low_1w[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily Donchian low or weekly trend fails
            if (close[i] < donchian_low[i] or 
                close[i] < donchian_low_1w[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily Donchian high or weekly trend fails
            if (close[i] > donchian_high[i] or 
                close[i] > donchian_high_1w[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals