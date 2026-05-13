#!/usr/bin/env python3
# 4h_Chaikin_Money_Flow_Trend_Filter
# Hypothesis: Chaikin Money Flow (CMF) measures institutional buying/selling pressure.
# In bull markets, buy when CMF > 0.15 + price > 50-period EMA + volume spike.
# In bear markets, sell when CMF < -0.15 + price < 50-period EMA + volume spike.
# Uses 1-day trend filter to avoid counter-trend trades. Designed for low-frequency,
# high-conviction trades with strong institutional confirmation.

name = "4h_Chaikin_Money_Flow_Trend_Filter"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Chaikin Money Flow (CMF) - 20 period
    # CMF = Sum((Close - Low - (High - Close)) / (High - Low) * Volume) / Sum(Volume)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # prevent div by zero
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume
    
    # Sum over 20 periods
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    
    # Avoid division by zero
    volume_sum = np.where(volume_sum == 0, 1e-10, volume_sum)
    cmf = mf_volume_sum / volume_sum
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(cmf[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # CMF conditions
        cmf_bullish = cmf[i] > 0.15
        cmf_bearish = cmf[i] < -0.15
        
        # Trend conditions
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Strong buying pressure + uptrend + volume spike
            if cmf_bullish and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong selling pressure + downtrend + volume spike
            elif cmf_bearish and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Selling pressure emerges OR trend breaks
            if cmf_bearish or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Buying pressure emerges OR trend breaks
            if cmf_bullish or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals