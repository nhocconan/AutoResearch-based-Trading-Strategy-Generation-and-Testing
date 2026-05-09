# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
- Price action: Breakout of Camarilla R1/S1 (daily) on 12h close
- Trend filter: 1d EMA34 (trend direction)
- Volume filter: 12h volume > 1.5x 20-period MA
- Position size: ±0.25
- Stop: exit when price crosses back through EMA34
- Timeframe: 12h
- Hypothesis: Camarilla levels act as institutional support/resistance; breakouts with volume and trend continuation yield sustained moves. Works in bull (breakouts up) and bear (breakouts down) as trend filter adapts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_width = 1.1 * (prev_high - prev_low) / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h volume confirmation: current volume > 1.5x 20-period MA
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: break above R1 with volume and above EMA34 (uptrend)
            if close[i] > r1_aligned[i] and vol_ok and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and below EMA34 (downtrend)
            elif close[i] < s1_aligned[i] and vol_ok and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34 (trend change)
            if close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 (trend change)
            if close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals