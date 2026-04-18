#!/usr/bin/env python3
"""
1d_Daily_Weekly_Pivot_Breakout_With_Trend_Filter
Hypothesis: Weekly pivot levels (R1, S1) act as strong support/resistance. Breakout above R1 with bullish weekly EMA trend = long; breakdown below S1 with bearish weekly EMA trend = short. Daily volume confirmation filters false breakouts. Designed for low frequency (10-25 trades/year) to minimize fee drag and work in both bull and bear markets via trend filter.
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
    
    # Weekly OHLC from higher timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot levels: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_p = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_p - weekly_low
    weekly_s1 = 2 * weekly_p - weekly_high
    
    # Align weekly levels to daily timeframe (wait for weekly bar close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly EMA trend filter (34-period)
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily volume filter: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        ema_trend = weekly_ema_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below weekly pivot point or trend reverses
            weekly_p_aligned = (weekly_high + weekly_low + weekly_close) / 3
            weekly_p_aligned = align_htf_to_ltf(prices, df_1w, weekly_p_aligned)
            if price < weekly_p_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above weekly pivot point or trend reverses
            weekly_p_aligned = (weekly_high + weekly_low + weekly_close) / 3
            weekly_p_aligned = align_htf_to_ltf(prices, df_1w, weekly_p_aligned)
            if price > weekly_p_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Daily_Weekly_Pivot_Breakout_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0