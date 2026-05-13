#!/usr/bin/env python3
# 4h_CCI_Overbought_Oversold_With_Volume_Filter
# Hypothesis: The Commodity Channel Index (CCI) identifies cyclical overbought/oversold conditions.
# In ranging markets, price tends to revert from extreme CCI levels (>100 or <-100).
# Trend strength is filtered using 4H ADX (<25 indicates ranging market).
# Volume confirmation ensures institutional participation during reversals.
# Works in both bull and bear markets by fading extremes in ranging conditions.

name = "4h_CCI_Overbought_Oversold_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # CCI calculation (20-period)
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)

    # ADX calculation (14-period) for ranging market filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
            minus_dm[i] = max(high[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx

    adx = calculate_adx(high, low, close, 14)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(cci[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CCI < -100 (oversold) + ranging market (ADX < 25) + volume filter
            if cci[i] < -100 and adx[i] < 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: CCI > 100 (overbought) + ranging market (ADX < 25) + volume filter
            elif cci[i] > 100 and adx[i] < 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI crosses above -50 (recovery from oversold) or ADX > 30 (trend developing)
            if cci[i] > -50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI crosses below 50 (decline from overbought) or ADX > 30 (trend developing)
            if cci[i] < 50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals