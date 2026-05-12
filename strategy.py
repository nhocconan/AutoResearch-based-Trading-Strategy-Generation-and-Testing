#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot breakout (R1/S1) on 12h timeframe with 1d trend filter and volume confirmation
# Works in bull markets by buying breakouts above R1 in uptrends, and in bear markets by selling breakdowns below S1 in downtrends
# Uses tight entry conditions to limit trades (~20-40/year) and avoid fee drag

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Pivot Points (1d) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align pivot levels to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # === Volume Confirmation (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h[i]) or np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1, above 1d EMA34, volume above average
            if (close[i] > R1_12h[i] and 
                close[i] > ema_34_12h[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, below 1d EMA34, volume above average
            elif (close[i] < S1_12h[i] and 
                  close[i] < ema_34_12h[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below S1 or trend change (below 1d EMA34)
            if (close[i] < S1_12h[i] or 
                close[i] < ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R1 or trend change (above 1d EMA34)
            if (close[i] > R1_12h[i] or 
                close[i] > ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals