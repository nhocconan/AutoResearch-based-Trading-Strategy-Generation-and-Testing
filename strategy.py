#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian AND close > 1w EMA50 AND volume > 1.5x average
# Short when price breaks below lower Donchian AND close < 1w EMA50 AND volume > 1.5x average
# Exit when price crosses Donchian middle (mean reversion) OR trend reversal (price crosses 1w EMA50)
# Uses 1d timeframe for lower trade frequency (~15-35/year) to minimize fee drag.
# Donchian provides structure, 1w EMA filters trend, volume confirms breakout strength.
# Works in bull via breakout continuation, bear via faded rallies and mean reversion in ranges.

name = "1d_Donchian20_1wEMA50_Volume_v1"
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
    
    # Get 1d data for Donchian calculation (already 1d, but use get_htf_data for consistency)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d data
    # Upper: 20-period high, Lower: 20-period low, Middle: average of upper/lower
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    middle_20 = (upper_20 + lower_20) / 2
    
    # Align Donchian levels to 1d timeframe (already aligned since calculated on 1d)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 1d volume > 1.5x 20-period average (spike confirmation)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and EMA
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper Donchian AND close > 1w EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower Donchian AND close < 1w EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle Donchian (mean reversion) OR trend reversal (close < 1w EMA50)
            if close[i] < middle_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle Donchian (mean reversion) OR trend reversal (close > 1w EMA50)
            if close[i] > middle_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals