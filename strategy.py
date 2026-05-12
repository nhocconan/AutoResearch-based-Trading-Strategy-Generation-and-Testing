#!/usr/bin/env python3
"""
1D_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER
Hypothesis: Daily chart breakout of Camarilla R1/S1 levels with weekly trend filter.
Only takes long when weekly EMA50 is rising and short when falling.
Uses volume spike to confirm breakout strength. Designed for ~10-20 trades/year on 1d.
Works in bull markets via long breakouts and bear via short breakdowns.
"""
name = "1D_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER"
timeframe = "1d"
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
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_r1[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 4
        camarilla_s1[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 4
    
    # Weekly trend filter: EMA50 slope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_slope = np.zeros_like(ema_50_1w_aligned)
    ema_slope[1:] = ema_50_1w_aligned[1:] - ema_50_1w_aligned[:-1]
    
    # Daily volume spike (20-day average)
    vol_ma_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / vol_ma_20d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike and rising weekly EMA
            if (high[i] > camarilla_r1[i] and 
                vol_spike[i] > 1.5 and 
                ema_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and falling weekly EMA
            elif (low[i] < camarilla_s1[i] and 
                  vol_spike[i] > 1.5 and 
                  ema_slope[i] < 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (mean reversion)
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (mean reversion)
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals