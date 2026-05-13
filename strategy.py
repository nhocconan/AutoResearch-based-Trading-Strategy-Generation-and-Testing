#!/usr/bin/env python3
# 6h_Chaikin_Money_Flow_Volume_Price_Confirmation
# Hypothesis: Chaikin Money Flow (CMF) on 12h for institutional flow direction, combined with 6h price action near VWAP and volume confirmation.
# CMF > 0 indicates buying pressure, CMF < 0 selling pressure. Enter long when CMF turns positive, price above VWAP, and volume spike.
# Enter short when CMF turns negative, price below VWAP, and volume spike. Uses 12h CMF to avoid noise, 6h for execution.
# Designed for 15-25 trades/year to minimize fee drift. Works in bull/bear: follows institutional flow.

name = "6h_Chaikin_Money_Flow_Volume_Price_Confirmation"
timeframe = "6h"
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

    # Get 12h data for CMF calculation (institutional flow)
    df_12h = get_htf_data(prices, '12h')

    # Calculate Chaikin Money Flow (CMF) on 12h
    # CMF = Sum((Close - Low - (High - Close)) / (High - Low) * Volume) / Sum(Volume) over period
    # Avoid division by zero
    hl_range = df_12h['high'] - df_12h['low']
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # prevent div by zero
    mf_multiplier = ((df_12h['close'] - df_12h['low']) - (df_12h['high'] - df_12h['close'])) / hl_range
    mf_volume = mf_multiplier * df_12h['volume']
    
    # 20-period CMF
    cmf_20 = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum() / \
             pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).sum()
    cmf_20_values = cmf_20.values

    # Align CMF to 6h timeframe (available after 12h bar closes)
    cmf_aligned = align_htf_to_ltf(prices, df_12h, cmf_20_values)

    # 6h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum()
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.values

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(cmf_aligned[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF turns positive, price above VWAP, volume spike
            if (cmf_aligned[i] > 0 and 
                cmf_aligned[i-1] <= 0 and  # crossed above zero
                close[i] > vwap[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF turns negative, price below VWAP, volume spike
            elif (cmf_aligned[i] < 0 and 
                  cmf_aligned[i-1] >= 0 and  # crossed below zero
                  close[i] < vwap[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF turns negative or price below VWAP
            if cmf_aligned[i] < 0 or close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF turns positive or price above VWAP
            if cmf_aligned[i] > 0 or close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals