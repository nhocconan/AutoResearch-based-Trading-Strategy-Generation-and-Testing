#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe with 1-day EMA trend filter and volume confirmation.
Trades breakouts of key intraday support/resistance levels in trending markets. Works in bull/bear by using EMA trend filter.
Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    def camarilla_levels(h, l, c):
        range_ = h - l
        if range_ <= 0:
            return c, c, c, c  # fallback
        R4 = c + (range_ * 1.1 / 2)
        R3 = c + (range_ * 1.1/4)
        R2 = c + (range_ * 1.1/6)
        R1 = c + (range_ * 1.1/12)
        S1 = c - (range_ * 1.1/12)
        S2 = c - (range_ * 1.1/6)
        S3 = c - (range_ * 1.1/4)
        S4 = c - (range_ * 1.1/2)
        return R1, R2, S1, S2
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    R1_1d = np.zeros(len(df_1d))
    R2_1d = np.zeros(len(df_1d))
    S1_1d = np.zeros(len(df_1d))
    S2_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        R1, R2, S1, S2 = camarilla_levels(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
        R1_1d[i] = R1
        R2_1d[i] = R2
        S1_1d[i] = S1
        S2_1d[i] = S2
    
    # Align to 12h timeframe
    R1_12h = align_ltf_to_htf(prices, df_1d, R1_1d)
    R2_12h = align_ltf_to_htf(prices, df_1d, R2_1d)
    S1_12h = align_ltf_to_htf(prices, df_1d, S1_1d)
    S2_12h = align_ltf_to_htf(prices, df_1d, S2_1d)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h = align_ltf_to_htf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_12h[i]) or np.isnan(R2_12h[i]) or 
            np.isnan(S1_12h[i]) or np.isnan(S2_12h[i]) or
            np.isnan(ema_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_12h[i]
        r2 = R2_12h[i]
        s1 = S1_12h[i]
        s2 = S2_12h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below S2 or trend reverses
            if price < s2 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above R2 or trend reverses
            if price > r2 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0