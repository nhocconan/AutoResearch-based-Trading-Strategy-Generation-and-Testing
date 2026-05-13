#!/usr/bin/env python3
# 12h_1W_Multiplier_Band_Trend_Follow
# Hypothesis: In strong trends, price remains outside the 1W Bollinger Bands (20, 2.0).
# Go long when price closes above upper band with volume confirmation.
# Go short when price closes below lower band with volume confirmation.
# Exit when price re-enters the bands or volatility drops.
# Uses 1W timeframe for trend context, 12h for entry/exit. Works in bull/bear by following higher timeframe volatility expansion/contraction.

name = "12h_1W_Multiplier_Band_Trend_Follow"
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

    # Get 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands on 1w close (20, 2.0)
    close_1w = pd.Series(df_1w['close'])
    sma_20 = close_1w.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1w.rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Align Bollinger Bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation: volume > 1.5 * 20-period average (~10 days at 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above upper band + volume spike
            if close[i] > upper_band_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower band + volume spike
            elif close[i] < lower_band_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters bands (below upper band)
            if close[i] < upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters bands (above lower band)
            if close[i] > lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals