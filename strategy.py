#!/usr/bin/env python3
# 12H_Daily_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade Daily Camarilla R1/S1 breakouts aligned with daily EMA34 trend and volume spikes.
# Long when: price breaks above R1 during daily uptrend (close > EMA34) with volume > 2x average.
# Short when: price breaks below S1 during daily downtrend (close < EMA34) with volume > 2x average.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 12-37 trades/year per symbol.

name = "12H_Daily_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Calculate Camarilla levels from previous daily bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's values (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid values (from roll), but we'll handle with min_periods later
    rng = prev_high - prev_low
    r1 = prev_close + rng * 1.1 / 12
    s1 = prev_close - rng * 1.1 / 12
    
    # Align daily data to 12h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R1 + volume spike
            if daily_up and volume_confirm:
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + price breaks below S1 + volume spike
            elif daily_down and volume_confirm:
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend breaks down or price closes below R1
            if not daily_up or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breaks up or price closes above S1
            if not daily_down or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals