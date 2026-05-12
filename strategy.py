#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_TrendVol_v1
Hypothesis: 4-hour breakouts from Camarilla R1/S1 levels (based on 12-hour price action) with 12-hour trend filter and volume spike confirmation.
This strategy targets 4h timeframe for optimal trade frequency (~30-50/year) while using 12h Camarilla levels for structure and 12h trend for filter.
Only takes long when price breaks above R1 with volume spike and 12h uptrend, short when breaks below S1 with volume spike and 12h downtrend.
Designed to work in both bull and bear markets via trend filter and volume confirmation to avoid false breakouts.
"""

name = "4h_12h_Camarilla_R1_S1_Breakout_TrendVol_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Using standard Camarilla formula based on previous bar's range
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Avoid look-ahead: only use previous bar's data
    range_ = prev_high - prev_low
    R1 = prev_close + 1.1 * range_ / 12
    S1 = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 12h EMA34
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 12h EMA34
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla range (between S1 and R1) OR closes below 12h EMA34
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla range (between S1 and R1) OR closes above 12h EMA34
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals