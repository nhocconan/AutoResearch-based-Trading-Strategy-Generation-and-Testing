#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Use weekly trend filter (EMA34) with daily Camarilla R1/S1 breakout and volume confirmation. Designed to capture breakouts in trending markets while avoiding chop. Weekly trend ensures we only trade in the direction of the higher timeframe trend, reducing false signals. Target 10-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Weekly trend filter: 34-period EMA ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R1 = np.zeros(len(df_1d))
    S1 = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        R1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
        S1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_trend = ema_34_1w_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 + weekly uptrend + volume spike
            if (price_close > r1 and 
                price_close > weekly_trend and 
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + weekly downtrend + volume spike
            elif (price_close < s1 and 
                  price_close < weekly_trend and 
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to previous day's close (mean reversion)
            prev_close = close_1d[i-1] if i-1 < len(close_1d) else close_1d[-1]
            if position == 1 and price_close < prev_close:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > prev_close:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0