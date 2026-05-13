#!/usr/bin/env python3
# 4h_200Day_EMA_Slope_Filtered_Breakout
# Hypothesis: The 200-day EMA slope acts as a strong trend filter across bull and bear markets.
# Enter long when price breaks above the rising 200-day EMA with volume confirmation and bullish momentum.
# Enter short when price breaks below the falling 200-day EMA with volume confirmation and bearish momentum.
# Exit when price crosses back below/above the 200-day EMA or momentum reverses.
# Uses 1d EMA200 slope to avoid whipsaws in sideways markets while capturing sustained trends.

name = "4h_200Day_EMA_Slope_Filtered_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA200 on 1d close
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA200 slope: positive slope = rising trend, negative slope = falling trend
    ema200_slope_1d = np.diff(ema200_1d, prepend=ema200_1d[0])
    
    # Align 1d EMA200 and its slope to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_slope_1d)
    
    # 4H momentum: EMA21 > EMA50 for bullish, EMA21 < EMA50 for bearish
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema200_slope_1d_aligned[i]) or 
            np.isnan(ema21[i]) or 
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above rising 200-day EMA + bullish 4H momentum + volume spike
            if (close[i] > ema200_1d_aligned[i] and 
                ema200_slope_1d_aligned[i] > 0 and 
                ema21[i] > ema50[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below falling 200-day EMA + bearish 4H momentum + volume spike
            elif (close[i] < ema200_1d_aligned[i] and 
                  ema200_slope_1d_aligned[i] < 0 and 
                  ema21[i] < ema50[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below 200-day EMA or momentum turns bearish
            if close[i] < ema200_1d_aligned[i] or ema21[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above 200-day EMA or momentum turns bullish
            if close[i] > ema200_1d_aligned[i] or ema21[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals