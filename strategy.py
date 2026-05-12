#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_TREND_VOLUME_SPIKE
# Hypothesis: Use 1d Camarilla R1/S1 levels as breakout triggers, confirmed by 1d EMA34 trend and volume spike (1.5x 20-period average). 
# In bull markets: buy breaks above R1 when above EMA34. In bear markets: sell breaks below S1 when below EMA34.
# Volume spike ensures institutional participation. Designed for low-frequency, high-conviction trades (target: 20-40/year).
# Works in both regimes: breakouts in trends, avoids false breakouts in chop via volume filter.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_TREND_VOLUME_SPIKE"
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
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > vol_ma * 1.5
    
    # 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero
    hl_range = prev_high - prev_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    r1 = prev_close + hl_range * 1.1 / 12
    s1 = prev_close - hl_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 with volume spike and uptrend (price > EMA34)
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with volume spike and downtrend (price < EMA34)
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns below R1 or trend breaks (price < EMA34)
            if close[i] < r1_aligned[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns above S1 or trend breaks (price > EMA34)
            if close[i] > s1_aligned[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals