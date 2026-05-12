#!/usr/bin/env python3
"""
12h_VWAP_MeanReversion_Range
Hypothesis: In ranging markets (choppy regime), price reverts to VWAP with high probability.
Long when price crosses above VWAP from below with volume confirmation, short when crosses below VWAP from above.
Uses 1d ADX < 25 to identify ranging regime and avoid trending markets. Targets 12-37 trades/year on 12h timeframe.
"""

name = "12h_VWAP_MeanReversion_Range"
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

    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    # Handle division by zero at start
    vwap = np.where(vwap_den == 0, typical_price, vwap)

    # Get 1d data for ADX ranging filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate ADX (14) on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(low)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            if up > down and up > 0:
                plus_dm[i] = up
            else:
                plus_dm[i] = 0
            if down > up and down > 0:
                minus_dm[i] = down
            else:
                minus_dm[i] = 0
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(low)
        
        atr[period-1] = np.mean(tr[1:period])
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx

    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        adx_val = adx_1d_aligned[i]
        vwap_val = vwap[i]
        vol_avg_10 = np.mean(volume[max(0, i-9):i+1]) if i >= 9 else np.mean(volume[:i+1])

        # Skip if ADX not available
        if np.isnan(adx_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range filter: only trade when ADX < 25 (ranging market)
        if adx_val >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above VWAP with volume confirmation
            if (close[i] > vwap_val and close[i-1] <= vwap[i-1] and 
                volume[i] > vol_avg_10 * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below VWAP with volume confirmation
            elif (close[i] < vwap_val and close[i-1] >= vwap[i-1] and 
                  volume[i] > vol_avg_10 * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below VWAP
            if close[i] < vwap_val and close[i-1] >= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above VWAP
            if close[i] > vwap_val and close[i-1] <= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals