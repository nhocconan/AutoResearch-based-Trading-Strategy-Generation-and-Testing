#!/usr/bin/env python3
# 4h_PriceChannel_Breakout_VolumeTrend
# Hypothesis: Breakouts at 4h Donchian(20) channels with 12h EMA200 trend filter and volume confirmation (1.5x 24-period average).
# Works in bull markets via long breakouts in uptrend and bear markets via short breakouts in downtrend.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.

name = "4h_PriceChannel_Breakout_VolumeTrend"
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
    
    # 12h trend filter (EMA200)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_12h_up = close_12h > ema200_12h
    trend_12h_down = close_12h < ema200_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with volume confirmation, 12h uptrend
            if (high[i] > high_20[i] and
                trend_12h_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume confirmation, 12h downtrend
            elif (low[i] < low_20[i] and
                  trend_12h_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 20-period low or 12h trend turns down
            if (low[i] < low_20[i] or
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 20-period high or 12h trend turns up
            if (high[i] > high_20[i] or
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals