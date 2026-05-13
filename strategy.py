#!/usr/bin/env python3
"""
6h_Volume_Weighted_CCI_Trend_Filter
Hypothesis: CCI identifies overbought/oversold conditions, but in strong trends it can remain extreme. 
We filter CCI extremes with volume-weighted price action and 12h EMA trend. 
Long when CCI < -100 and price closes above VWAP with rising volume, short when CCI > 100 and price closes below VWAP with falling volume.
Exit on CCI crossing zero or trend reversal. Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend).
Target: 20-60 trades/year to minimize fee drag.
"""

name = "6h_Volume_Weighted_CCI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # CCI calculation (20-period)
    tp = typical_price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume-weighted price change: positive if typical price > VWAP and volume increasing
    price_vwap_diff = typical_price - vwap
    vol_change = np.diff(volume, prepend=volume[0])
    vol_weighted_signal = price_vwap_diff * vol_change
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after CCI warmup
        if position == 0:
            # LONG: CCI oversold (< -100) AND price above VWAP with rising volume AND uptrend
            if (cci[i] < -100 and 
                price_vwap_diff[i] > 0 and 
                vol_change[i] > 0 and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CCI overbought (> 100) AND price below VWAP with falling volume AND downtrend
            elif (cci[i] > 100 and 
                  price_vwap_diff[i] < 0 and 
                  vol_change[i] < 0 and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI crosses above zero OR trend reverses
            if (cci[i] > 0) or \
               (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI crosses below zero OR trend reverses
            if (cci[i] < 0) or \
               (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals