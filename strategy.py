#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) on 4h timeframe act as strong support/resistance.
Breakouts above R1 or below S1 on 1h with volume confirmation and 4h trend alignment capture
momentum moves while avoiding false breakouts. Exit on reversion to the pivot point (PP).
Position size 0.20 targets ~15-25 trades/year per symbol to minimize fee drag.
Works in bull/bear markets via trend filter and volatility-based volume confirmation.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for Camarilla pivot calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels for each 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    camarilla_pp = (h_4h + l_4h + c_4h) / 3.0
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12.0
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h chart
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average (10 hours on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (reduces noise outside active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation and uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation and downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or trend reverses
            if (close[i] < camarilla_pp_aligned[i]) or \
               (close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or trend reverses
            if (close[i] > camarilla_pp_aligned[i]) or \
               (close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals