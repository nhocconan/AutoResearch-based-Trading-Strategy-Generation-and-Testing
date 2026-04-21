#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot level (R1/S1) breakout with volume confirmation and 12h EMA trend filter.
Camarilla levels act as intraday support/resistance; breakouts indicate momentum shifts.
Volume confirms institutional participation. 12h EMA filter ensures alignment with higher timeframe trend.
Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + range_1d * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12.0
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        ema_trend = ema_12h_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + volume confirmation + price > 12h EMA (bullish alignment)
            if (price_close > r1[i] and 
                vol_ratio_val > 1.5 and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + volume confirmation + price < 12h EMA (bearish alignment)
            elif (price_close < s1[i] and 
                  vol_ratio_val > 1.5 and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to pivot point (PP) or volume drops
            if position == 1 and (price_close < pp[i] or vol_ratio_val < 1.0):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > pp[i] or vol_ratio_val < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0