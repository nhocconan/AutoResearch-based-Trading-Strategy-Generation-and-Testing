#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w trend filter.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d volume > 1.5 * 20-period average volume AND 1w close > 1w EMA50.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d volume > 1.5 * 20-period average volume AND 1w close < 1w EMA50.
# Exit when Alligator lines re-cross (jaws-teeth or teeth-lips crossover).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 12h timeframe with strict entry conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h.

name = "12h_WilliamsAlligator_1dVolumeConfirm_1wTrendFilter_v1"
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
    
    # Calculate Williams Alligator on 12h data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaws = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to LTF (12h)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w = close_1w > ema_50_1w  # True for bullish, False for bearish
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish alignment (jaws < teeth < lips) AND volume confirmation AND 1w bullish trend
            if (jaws_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and
                volume_confirm_1d_aligned[i] > 0.5 and trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish alignment (jaws > teeth > lips) AND volume confirmation AND 1w bearish trend
            elif (jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                  volume_confirm_1d_aligned[i] > 0.5 and not trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator re-cross (jaws-teeth or teeth-lips crossover)
            if (jaws_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator re-cross (jaws-teeth or teeth-lips crossover)
            if (jaws_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals