#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day average volume
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day average volume
# Exit when price crosses the 10-day moving average in opposite direction OR Alligator alignment breaks
# Uses 1d timeframe for lower frequency (~15-35 trades/year), Donchian for structure, 1w EMA for trend filter, volume for confirmation.
# Target: 60-140 total trades over 4 years (15-35/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1d_Donchian20_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Donchian channels on 1d: 20-period high/low, 10-period exit
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ma_10[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > 20-day high AND price > 1w EMA50 AND volume confirmation
            if close[i] > high_20[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < 20-day low AND price < 1w EMA50 AND volume confirmation
            elif close[i] < low_20[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < 10-day MA OR price < 1w EMA50 (trend failure)
            if close[i] < ma_10[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > 10-day MA OR price > 1w EMA50 (trend failure)
            if close[i] > ma_10[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals