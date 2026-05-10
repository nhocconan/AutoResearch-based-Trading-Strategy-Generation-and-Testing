#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade breakouts of Camarilla R1/S1 levels in direction of 1d trend with volume confirmation.
# Long when: price breaks above R1, 1d EMA50 uptrend, volume > 1.5x average.
# Short when: price breaks below S1, 1d EMA50 downtrend, volume > 1.5x average.
# Exit when: price returns to Camarilla Pivot point.
# Uses volume confirmation to filter false breakouts and trend alignment to work in bull/bear.
# Target: 20-40 trades/year per symbol.

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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    # We'll use daily data to calculate these
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Pivot = (H + L + C) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align daily levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_4h = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    daily_uptrend_4h = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_4h = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(Pivot_4h[i]) or
            np.isnan(daily_uptrend_4h[i]) or np.isnan(daily_downtrend_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_4h[i] > 0.5
        daily_down = daily_downtrend_4h[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above R1, daily uptrend, volume confirmation
            if daily_up and volume_confirm and close[i] > R1_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, daily downtrend, volume confirmation
            elif daily_down and volume_confirm and close[i] < S1_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit when price returns to Pivot
            if close[i] < Pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit when price returns to Pivot
            if close[i] > Pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals