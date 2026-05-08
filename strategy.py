# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Buy when price breaks above upper Donchian channel in 1w uptrend with volume spike.
# Sell when price breaks below lower Donchian channel in 1w downtrend with volume spike.
# Uses 1w EMA(21) for trend and 30-period volume spike for confirmation.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag.
# Works in both bull (trend following) and bear (counter-trend at extremes) markets.

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(21) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_up = ema_21_1w[1:] > ema_21_1w[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1w index
    
    # Donchian channels (20-period) on 1d data
    # Upper channel: highest high of past 20 periods
    # Lower channel: lowest low of past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 30-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1w trend to 1d timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for Donchian and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above upper channel in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1w uptrend
                close[i] >= upper_channel[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower channel in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1w downtrend
                  close[i] <= lower_channel[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal (break below lower channel)
            if (trend_up_aligned[i] <= 0.5 and  # 1w downtrend
                close[i] <= lower_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal (break above upper channel)
            if (trend_up_aligned[i] > 0.5 and  # 1w uptrend
                  close[i] >= upper_channel[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals