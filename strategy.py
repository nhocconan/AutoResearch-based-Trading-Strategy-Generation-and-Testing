#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = Low - EMA13(close)
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA34 AND volume > 1.5x average
# Exit when momentum diverges (Bull Power <= 0 for long, Bear Power >= 0 for short) OR trend reversal
# Uses 6h timeframe for lower frequency, Elder Ray for momentum strength, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via momentum continuation, bear via faded rallies.

name = "6h_ElderRay_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA(13) on 6h close for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13; Bear Power = Low - EMA13
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 AND volume confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema34_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA34 AND volume confirmation
            elif bear_power[i] < 0 and bull_power[i] < 0 and close[i] < ema34_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (momentum fading) OR trend reversal (price < 1d EMA34)
            if bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (momentum fading) OR trend reversal (price > 1d EMA34)
            if bear_power[i] >= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals