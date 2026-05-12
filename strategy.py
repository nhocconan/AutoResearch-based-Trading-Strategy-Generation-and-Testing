#!/usr/bin/env python3
"""
4h_Supertrend_1dVWAP_MeanReversion
Hypothesis: In mean-reverting markets (BTC/ETH), price reverts to VWAP after extreme deviations. Supertrend on 4h determines regime: long when price < VWAP in uptrend, short when price > VWAP in downtrend. Uses 1d VWAP for institutional reference and avoids overtrading with strict entry conditions. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

name = "4h_Supertrend_1dVWAP_MeanReversion"
timeframe = "4h"
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
    
    # Supertrend on 4h: ATR(10), factor=3.0
    atr_period = 10
    factor = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # Initialize bands
    upper_band_final = np.full_like(upper_band, np.nan)
    lower_band_final = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close, np.nan)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    upper_band_final[0] = upper_band[0]
    lower_band_final[0] = lower_band[0]
    supertrend[0] = upper_band_final[0]
    trend[0] = 1
    
    for i in range(1, n):
        if close[i-1] > upper_band_final[i-1]:
            trend[i] = -1
        elif close[i-1] < lower_band_final[i-1]:
            trend[i] = 1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            upper_band_final[i] = min(upper_band[i], upper_band_final[i-1])
            lower_band_final[i] = lower_band[i]
            supertrend[i] = lower_band_final[i]
        else:
            upper_band_final[i] = upper_band[i]
            lower_band_final[i] = max(lower_band[i], lower_band_final[i-1])
            supertrend[i] = upper_band_final[i]
    
    # 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Typical price and VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align 1d VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Volume filter: >1.5x 20-period average (avoid chop)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(supertrend[i]) or 
            np.isnan(vwap_aligned[i]) or
            np.isnan(trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price below VWAP in uptrend (buy the dip)
            if (close[i] < vwap_aligned[i] and 
                trend[i] == 1 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP in downtrend (sell the rally)
            elif (close[i] > vwap_aligned[i] and 
                  trend[i] == -1 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP or trend turns down
            if close[i] > vwap_aligned[i] or trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP or trend turns up
            if close[i] < vwap_aligned[i] or trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals