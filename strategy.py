#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Trade breakouts of Camarilla R3/S3 levels from 12h price action with 1d trend filter and volume spike.
# Long when: price breaks above R3 (bullish breakout) with 1d uptrend and volume > 2x average.
# Short when: price breaks below S3 (bearish breakdown) with 1d downtrend and volume > 2x average.
# Uses Camarilla levels calculated from prior 12h bar's high-low-close for current bar.
# Works in bull/bear by following 1d trend direction and using volume to confirm institutional participation.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate Camarilla levels for each bar based on PREVIOUS bar's HLC
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Shift by 1 to use previous bar's data for current bar's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    hl_range = prev_high - prev_low
    r3 = prev_close + hl_range * 1.1 / 4
    s3 = prev_close - hl_range * 1.1 / 4
    
    # Volume spike detection: volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    volume_spike = vol_ratio > 2.0
    
    # Daily trend filter from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA on 1d close
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(volume_spike[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + 1d uptrend + volume spike
            if close[i] > r3[i] and daily_uptrend_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + 1d downtrend + volume spike
            elif close[i] < s3[i] and daily_downtrend_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or trend changes
            if close[i] < s3[i] or daily_uptrend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or trend changes
            if close[i] > r3[i] or daily_downtrend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals