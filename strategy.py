#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade breakouts from Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
# Long when: price breaks above R1 with 1d uptrend and volume > 1.5x average.
# Short when: price breaks below S1 with 1d downtrend and volume > 1.5x average.
# Uses 12h timeframe for lower frequency (12-37 trades/year target) to minimize fee drag.
# Works in bull/bear by following 1d trend and using volume to confirm institutional interest.

name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 12h price for Camarilla calculation (use previous bar's HLC)
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Camarilla levels calculation
    range_hl = prev_high - prev_low
    R1 = prev_close + range_hl * 1.1 / 12
    S1 = prev_close - range_hl * 1.1 / 12
    
    # Volume average (24-period for 12h = 2 days)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R1 + volume
            if daily_up and volume_confirm and close[i] > R1[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S1 + volume
            elif daily_down and volume_confirm and close[i] < S1[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend reverses or price drops below R1
            if not daily_up or close[i] < R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reverses or price rises above S1
            if not daily_down or close[i] > S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals