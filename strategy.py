#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
# Hypothesis: Uses weekly Camarilla R1/S1 levels filtered by weekly trend (EMA34) and volume spikes.
# Weekly trend filter reduces whipsaws; volume confirms breakout strength.
# Target: 15-35 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
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
    
    # Get weekly data for Camarilla pivot calculation (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels: R1, S1
    camarilla_range = high_1w - low_1w
    r1 = close_1w + 1.1 * camarilla_range / 12
    s1 = close_1w - 1.1 * camarilla_range / 12
    
    # Get weekly data for trend filter (EMA34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    ema_34_1w_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter on 12h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_1w_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > R1, above weekly EMA34 trend, volume spike
            if close[i] > r1_12h[i] and close[i] > ema_34_1w_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1, below weekly EMA34 trend, volume spike
            elif close[i] < s1_12h[i] and close[i] < ema_34_1w_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < R1 or below weekly EMA34 trend
            if close[i] < r1_12h[i] or close[i] < ema_34_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > S1 or above weekly EMA34 trend
            if close[i] > s1_12h[i] or close[i] > ema_34_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals