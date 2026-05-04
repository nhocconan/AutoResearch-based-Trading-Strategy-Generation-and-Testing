#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide clear breakout levels. Breakout above upper or below lower channel
# with volume confirms institutional participation. 1w EMA50 ensures alignment with long-term trend
# to avoid counter-trend trades. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakdowns in downtrend.

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND 1w uptrend AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_aligned[i] and  # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below lower channel AND 1w downtrend AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_aligned[i] and  # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower channel OR 1w trend turns down
            if close[i] < lower_channel[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above upper channel OR 1w trend turns up
            if close[i] > upper_channel[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals