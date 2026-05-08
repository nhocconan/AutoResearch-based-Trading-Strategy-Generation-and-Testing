#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high in 1d uptrend with volume spike.
# Short when price breaks below Donchian(20) low in 1d downtrend with volume spike.
# Uses 1d EMA(34) for trend and 60-period volume spike for confirmation.
# Target: 100-150 total trades over 4 years (25-38/year) to minimize fee drag.

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Donchian channel on 12h data
    window = 20
    donchian_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
    donchian_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
    
    # Volume confirmation: 60-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=60, adjust=False, min_periods=60).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1d indicators to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: breakout above Donchian high in uptrend
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= donchian_high[i] and
                vol_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: breakdown below Donchian low in downtrend
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= donchian_low[i] and
                  vol_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: reverse signal or stop
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] <= donchian_low[i]):  # Break below Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: reverse signal or stop
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= donchian_high[i]):  # Break above Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals