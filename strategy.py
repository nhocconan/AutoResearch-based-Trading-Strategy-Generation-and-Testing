#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1dVWAP_Support_Resistance
Hypothesis: KAMA on 4h adapts to market noise, providing reliable trend direction in both bull and bear markets.
Price returning to 1d VWAP acts as dynamic support/resistance. Long when price > KAMA and bouncing from 1d VWAP support;
short when price < KAMA and rejected from 1d VWAP resistance. Volume confirmation filters low-conviction moves.
Position size 0.25 targets ~20-30 trades/year to minimize fee drag.
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
    
    # KAMA on 4h: adapts to market noise (ER = Efficiency Ratio)
    # ER = |net change| / sum(|abs change|) over period
    # Smooth constant = [ER * (fastest - slowest) + slowest]^2
    kama_period = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close, k=10))  # 10-period net change
    abs_changes = np.abs(np.diff(close, k=1))    # 1-period changes
    volatility_sum = np.convolve(abs_changes, np.ones(kama_period), 'same')  # sum of abs changes
    volatility_sum[:kama_period-1] = np.cumsum(abs_changes)[:kama_period-1]  # fill beginning
    
    er = np.zeros_like(close)
    er[kama_period-1:] = price_change[kama_period-1:] / volatility_sum[kama_period-1:]
    er[volatility_sum == 0] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d VWAP calculation (typical price * volume) / cumulative volume
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    # Align 1d VWAP to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:19] = np.cumsum(volume)[:19] / np.arange(1, 20)
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: Price > KAMA and bouncing from 1d VWAP support (close > VWAP and low touched VWAP)
            if (close[i] > kama[i] and 
                close[i] > vwap_1d_aligned[i] and 
                low[i] <= vwap_1d_aligned[i] * 1.002 and  # allow small buffer
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA and rejected from 1d VWAP resistance (close < VWAP and high touched VWAP)
            elif (close[i] < kama[i] and 
                  close[i] < vwap_1d_aligned[i] and 
                  high[i] >= vwap_1d_aligned[i] * 0.998 and  # allow small buffer
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or breaks below VWAP support
            if close[i] < kama[i] or close[i] < vwap_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or breaks above VWAP resistance
            if close[i] > kama[i] or close[i] > vwap_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals