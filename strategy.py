#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: Camarilla pivot points (R1/S1) on 1h chart act as intraday support/resistance.
Breakouts above R1 or below S1 with 4h trend alignment and 1d volume confirmation capture momentum.
Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes via trend filter).
Uses 4h for trend direction, 1d for volume filter, 1h for precise entry timing to target 15-30 trades/year.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA34 for trend
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla pivot points on 1h chart
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close + (high - low) * 1.1 / 12.0
    camarilla_s1 = close - (high - low) * 1.1 / 12.0
    camarilla_pp = (high + low + close) / 3.0  # Pivot point for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Breakout above Camarilla R1 with 4h uptrend and 1d volume confirmation
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below Camarilla S1 with 4h downtrend and 1d volume confirmation
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Camarilla pivot point or 4h trend reverses
            if (close[i] < camarilla_pp[i]) or \
               (close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to Camarilla pivot point or 4h trend reverses
            if (close[i] > camarilla_pp[i]) or \
               (close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals