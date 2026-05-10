#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as key support/resistance.
# Breakouts above R1 or below S1 with volume spike and aligned daily EMA34 trend capture
# institutional moves. Works in bull (breakouts continue) and bear (false breakdowns reverse).
# Volume filter ensures participation; EMA34 filter avoids counter-trend whipsaws.
# Target: 20-40 trades/year to minimize fee drag.

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using (H-L) range, Camarilla R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    # Where C = (H+L+CLOSE)/3 of previous day
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Pivot point (typical price)
    pivot = (phigh + plow + pclose) / 3.0
    # Camarilla width
    rang = phigh - plow
    r1 = pivot + 1.1 * rang / 12.0
    s1 = pivot - 1.1 * rang / 12.0
    
    # Align Camarilla levels to 4h (previous day's levels available after daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    pclose_series = pd.Series(pclose)
    ema34 = pclose_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 5  # Need EMA34 and vol MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: break above R1 with volume and above daily EMA34 (bullish bias)
            if close[i] > r1_aligned[i] and vol_confirm and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and below daily EMA34 (bearish bias)
            elif close[i] < s1_aligned[i] and vol_confirm and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (invalidates bullish bias) or volume dries up
            if close[i] < s1_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 (invalidates bearish bias) or volume dries up
            if close[i] > r1_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals