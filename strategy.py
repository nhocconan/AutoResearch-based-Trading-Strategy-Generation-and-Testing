#!/usr/bin/env python3
"""
1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_1D_VOLUME
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA trend filter and daily volume confirmation. Uses 4h for signal direction (trend) and 1h for entry timing to reduce false breakouts. Daily volume filter ensures institutional participation. Target: 15-30 trades/year per symbol.
"""
name = "1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_1D_VOLUME"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (pre-market to post-close)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d data for Camarilla levels (using previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    ph = np.roll(high_1d, 1)  # previous day high
    pl = np.roll(low_1d, 1)   # previous day low
    pc = np.roll(close_1d, 1) # previous day close
    ph[0] = pl[0] = pc[0] = np.nan  # first day has no previous day
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily volume filter: current day volume > 1.5x 20-day average
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > 1.5 * vol_ma_1d
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume filter and price above 4h EMA34
            if (high[i] > r1_aligned[i] and 
                volume_filter_aligned[i] and 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume filter and price below 4h EMA34
            elif (low[i] < s1_aligned[i] and 
                  volume_filter_aligned[i] and 
                  close[i] < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S1 or below 4h EMA34
            if (close[i] < s1_aligned[i] or 
                close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rises back above R1 or above 4h EMA34
            if (close[i] > r1_aligned[i] or 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals