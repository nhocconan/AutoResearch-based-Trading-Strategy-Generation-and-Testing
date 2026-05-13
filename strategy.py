#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as key support/resistance levels. 
Breakouts above R1 or below S1 with volume confirmation and daily EMA34 trend filter capture 
trends in both bull and bear markets. Designed for low trade frequency (20-40/year) with 
clear entry/exit rules to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's values, so shift by 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_range = (prev_high - prev_low) * 1.1 / 12
    r1_levels = prev_close + camarilla_range
    s1_levels = prev_close - camarilla_range
    
    # Align daily levels to 4h timeframe (will be available after daily bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_levels)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend filter
            if close[i] > r1_aligned[i] and volume_confirm[i]:
                # Additional filter: only take long if price above daily EMA34 (uptrend)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend filter
            elif close[i] < s1_aligned[i] and volume_confirm[i]:
                # Additional filter: only take short if price below daily EMA34 (downtrend)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend changes (below EMA34)
            if close[i] < r1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend changes (above EMA34)
            if close[i] > s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals