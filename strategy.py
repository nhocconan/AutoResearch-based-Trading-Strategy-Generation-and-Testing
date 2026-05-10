#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses 1-day trend and volume spike to confirm 4-hour Camarilla breakouts
# Combines price channel breakout (Camarilla R1/S1) with institutional trend filtering
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Calculate 1-day volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(volume_1d)
    vol_sum = 0
    for i in range(len(volume_1d)):
        vol_sum += volume_1d[i]
        if i >= 20:
            vol_sum -= volume_1d[i-20]
        if i >= 19:
            vol_ma_1d[i] = vol_sum / 20
        else:
            vol_ma_1d[i] = np.nan
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)  # 2x volume spike
    
    # Align 1d indicators to 4h timeframe
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous bar values
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels (primary levels)
    range_4h = prev_high - prev_low
    R1 = prev_close + range_4h * 1.1 / 12
    S1 = prev_close - range_4h * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (they're already 4h, just need to shift for previous bar)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 1d uptrend
            if (high[i] > R1_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and 1d downtrend
            elif (low[i] < S1_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or 1d trend turns down
            if (low[i] < S1_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or 1d trend turns up
            if (high[i] > R1_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals