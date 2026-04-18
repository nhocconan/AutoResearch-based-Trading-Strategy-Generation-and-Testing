#!/usr/bin/env python3
"""
12h_Price_Action_With_Trend_Filter
Hypothesis: Price action at daily pivot levels (R1, S1) with 1d trend filter yields high-probability entries. 12h timeframe reduces noise while capturing multi-day moves. Works in bull/bear via trend filter and low frequency (15-30 trades/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily typical price for pivot calculation
    daily_tp = (high + low + close) / 3
    
    # Daily OHLC from higher timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily pivot levels: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_p = (daily_high + daily_low + daily_close) / 3
    daily_r1 = 2 * daily_p - daily_low
    daily_s1 = 2 * daily_p - daily_high
    
    # Align daily levels to 12h timeframe (wait for daily bar close)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_p_aligned = align_htf_to_ltf(prices, df_1d, daily_p)
    
    # Daily EMA trend filter (34-period)
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # 12h volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or 
            np.isnan(daily_ema_aligned[i]) or np.isnan(daily_p_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = daily_r1_aligned[i]
        s1 = daily_s1_aligned[i]
        p = daily_p_aligned[i]
        ema_trend = daily_ema_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above daily R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below daily pivot or trend reverses
            if price < p or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above daily pivot or trend reverses
            if price > p or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Price_Action_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0