#!/usr/bin/env python3
# 4H_WILLIAMS_ALLIGATOR_1D_TREND_VOLUME_CONFIRMATION
# Hypothesis: Williams Alligator (3 SMAs) combined with daily trend and volume confirmation captures trends in both bull and bear markets.
# Uses Alligator's jaw/teeth/lips alignment for trend direction, filters by 1d EMA50 trend, and requires volume spike (1.5x 20-bar avg).
# Target: 15-25 trades/year (60-100 total) to minimize fee drag while maintaining edge.

name = "4H_WILLIAMS_ALLIGATOR_1D_TREND_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: 13, 8, 5 period SMAs with future shift
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: 1.5x 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + daily uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + daily downtrend + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment (Lips < Teeth < Jaw)
            if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment (Lips > Teeth > Jaw)
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals