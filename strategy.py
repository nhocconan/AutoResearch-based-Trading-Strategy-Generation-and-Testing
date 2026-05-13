#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1dVWAP_Support_Resistance
Hypothesis: KAMA adapts to market noise, reducing whipsaw in ranging markets.
Trades in direction of KAMA trend when price is near 1d VWAP support/resistance
with volume confirmation. VWAP acts as dynamic support/resistance in trending
markets. Position size 0.25 limits risk and targets ~20-30 trades/year.
Works in bull (trend following) and bear (mean reversion near VWAP).
"""

name = "4h_KAMA_Trend_With_1dVWAP_Support_Resistance"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    
    # Align VWAP to 4h chart
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price above KAMA and near VWAP support with volume
            if (close[i] > kama[i] and 
                close[i] <= vwap_1d_aligned[i] * 1.01 and  # within 1% above VWAP
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and near VWAP resistance with volume
            elif (close[i] < kama[i] and 
                  close[i] >= vwap_1d_aligned[i] * 0.99 and  # within 1% below VWAP
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or moves far from VWAP
            if (close[i] < kama[i]) or (close[i] > vwap_1d_aligned[i] * 1.02):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or moves far from VWAP
            if (close[i] > kama[i]) or (close[i] < vwap_1d_aligned[i] * 0.98):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals