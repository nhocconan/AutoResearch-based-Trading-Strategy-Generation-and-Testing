# -*- coding: utf-8 -*-
# -*- mode: python -*-
#!/usr/bin/env python3

# 4H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade breakouts at Camarilla R1/S1 levels using 1d trend filter and volume confirmation.
# Long when: price breaks above R1, 1d EMA34 trend up, volume > 2x average.
# Short when: price breaks below S1, 1d EMA34 trend down, volume > 2x average.
# Exit when: price retouches the Camarilla Pivot (central level).
# Target: 20-35 trades/year per symbol. Works in bull/bear by following 1d trend.

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
    
    # 4h price data
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    Pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_4h = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_4h = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_4h = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(Pivot_4h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(trend_1d_up_4h[i]) or np.isnan(trend_1d_down_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        trend_up = trend_1d_up_4h[i] > 0.5
        trend_down = trend_1d_down_4h[i] > 0.5
        
        if position == 0:
            # Enter long: break above R1 + 1d uptrend + volume
            if close[i] > R1_4h[i] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 + 1d downtrend + volume
            elif close[i] < S1_4h[i] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price retouches Pivot (mean reversion to center)
            if close[i] <= Pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price retouches Pivot
            if close[i] >= Pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals