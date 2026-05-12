#!/usr/bin/env python3
"""
1d_VWAP_Reversion_with_Volume_Confirmation
Hypothesis: In 1d timeframe, price reverts to VWAP after deviation, especially when volume confirms the move. 
Long when price < VWAP - 1.5*std and volume > 1.5x average; short when price > VWAP + 1.5*std and volume > 1.5x average.
Exit when price returns to VWAP. Uses weekly trend filter to align with higher timeframe momentum.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_VWAP_Reversion_with_Volume_Confirmation"
timeframe = "1d"
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
    
    # Calculate VWAP and standard deviation of price deviation
    typical_price = (high + low + close) / 3
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    # Calculate rolling standard deviation of price deviation from VWAP
    price_dev = typical_price - vwap
    dev_std = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(vwap[i]) or
            np.isnan(dev_std[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price below VWAP - 1.5*std + volume spike + weekly uptrend
            if (typical_price[i] < vwap[i] - 1.5 * dev_std[i] and 
                volume_spike[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP + 1.5*std + volume spike + weekly downtrend
            elif (typical_price[i] > vwap[i] + 1.5 * dev_std[i] and 
                  volume_spike[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP
            if typical_price[i] >= vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP
            if typical_price[i] <= vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals