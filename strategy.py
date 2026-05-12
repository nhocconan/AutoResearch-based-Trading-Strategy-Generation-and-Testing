#!/usr/bin/env python3
"""
4h_VolumeWeighted_Camarilla_R1S1_Breakout
Hypothesis: Combines volume-weighted price action with tighter confirmation
to reduce false breakouts. Uses volume-weighted average price (VWAP) over
the last 20 periods as dynamic filter, requiring price to be above/below
VWAP by a margin proportional to ATR. This adds confluence without
excessive conditions, targeting 25-35 trades/year per symbol. Designed
to work in both trending and ranging markets by adapting to volatility.
"""

name = "4h_VolumeWeighted_Camarilla_R1S1_Breakout"
timeframe = "4h"
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
    
    # ATR for dynamic threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume-weighted average price (VWAP) over 20 periods
    vwap_num = np.zeros(n)
    vwap_den = np.zeros(n)
    for i in range(n):
        start = max(0, i - 19)
        vwap_num[i] = np.sum(close[start:i+1] * volume[start:i+1])
        vwap_den[i] = np.sum(volume[start:i+1])
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(close, np.nan), where=vwap_den!=0)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND is above VWAP by at least 0.5*ATR
            if (close[i] > camarilla_r1_aligned[i] and
                close[i] > vwap[i] + 0.5 * atr[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 AND is below VWAP by at least 0.5*ATR
            elif (close[i] < camarilla_s1_aligned[i] and
                  close[i] < vwap[i] - 0.5 * atr[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR falls below VWAP
            if (close[i] < camarilla_s1_aligned[i]) or \
               (close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR rises above VWAP
            if (close[i] > camarilla_r1_aligned[i]) or \
               (close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals