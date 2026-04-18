#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and 1d Trend Filter
Hypothesis: In 12h timeframe, Camarilla pivot levels (R1/S1) derived from 1d high/low/close
act as strong support/resistance. Breakouts above R1 or below S1 with volume confirmation
and alignment with 1d EMA34 trend capture genuine momentum moves. The 1d trend filter
avoids counter-trend trades, reducing false breakouts. Volume ensures participation.
Target: 15-30 trades/year to minimize fee drag on 12h timeframe.
"""

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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (using close)
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_r1 = close_1d_arr + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d_arr - 1.1 * (high_1d - low_1d) / 12
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: 1.5x 30-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_34 = ema_34_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and uptrend (price > EMA34 on 1d)
            if price > r1 and vol_ok and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and downtrend (price < EMA34 on 1d)
            elif price < s1 and vol_ok and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to S1 or trend changes
            if price < s1 or price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to R1 or trend changes
            if price > r1 or price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0