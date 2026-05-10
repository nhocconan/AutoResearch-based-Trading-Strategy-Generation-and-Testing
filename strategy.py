#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout at Camarilla R1/S1 levels with daily trend filter and volume confirmation.
# Long when: price breaks above R1, daily trend up (close > EMA34), volume > 1.5x average.
# Short when: price breaks below S1, daily trend down (close < EMA34), volume > 1.5x average.
# Uses 4-hour timeframe with 1-day trend filter for alignment across market regimes.
# Target: 20-30 trades/year per symbol to avoid fee drag.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Previous period high, low, close for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels calculation
    range_val = prev_high - prev_low
    R1 = prev_close + range_val * 1.1 / 12
    S1 = prev_close - range_val * 1.1 / 12
    
    # Volume confirmation
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align daily trend to 4h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: break above R1 + daily uptrend + volume confirmation
            if daily_up and volume_confirm and close[i] > R1[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 + daily downtrend + volume confirmation
            elif daily_down and volume_confirm and close[i] < S1[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to mean (pivot) or trend reverses
            pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            if close[i] < pivot or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to mean (pivot) or trend reverses
            pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            if close[i] > pivot or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals