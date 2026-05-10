#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Trade Camarilla pivot breakouts on 12h timeframe with 1d trend and volume confirmation.
# Long when: price breaks above R1 with 1d uptrend and volume > 1.3x average.
# Short when: price breaks below S1 with 1d downtrend and volume > 1.3x average.
# Uses volume confirmation to filter false breakouts and trend alignment to avoid counter-trend trades.
# Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull and bear markets.

name = "12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # 12h range calculation (using previous bar's high/low for Camarilla)
    # Shift by 1 to avoid look-ahead: use previous bar's range
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Camarilla levels for intraday trading
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_val = prev_high - prev_low
    r1 = prev_close + range_val * 1.1 / 12
    s1 = prev_close - range_val * 1.1 / 12
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Use EMA34 for trend (more responsive than EMA50 for 1d)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above R1 with 1d uptrend and volume confirmation
            if daily_up and volume_confirm and close[i] > r1[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with 1d downtrend and volume confirmation
            elif daily_down and volume_confirm and close[i] < s1[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves back below R1 or trend changes
            if close[i] < r1[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back above S1 or trend changes
            if close[i] > s1[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals