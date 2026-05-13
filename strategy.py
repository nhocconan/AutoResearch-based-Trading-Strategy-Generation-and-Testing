#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_Filter
Hypothesis: Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets.
In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), trade with the trend using
Williams %R overbought/oversold as entry timing. Uses 1-week trend filter and volume confirmation to reduce false signals.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag. Works in both bull and bear regimes
by only taking trades when Alligator is 'awake' (trending) and price aligns with higher timeframe trend.
"""

name = "4h_Williams_Alligator_Trend_Filter"
timeframe = "4h"
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
    
    # Get weekly data for 1-week trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Alligator conditions: JAW > TEETH > LIPS (uptrend), JAW < TEETH < LIPS (downtrend)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        is_uptrend = jaw_val > teeth_val > lips_val
        is_downtrend = jaw_val < teeth_val < lips_val
        
        if position == 0:
            # LONG: Alligator uptrend, Williams %R oversold (< -80), volume spike, above 1-week EMA50
            if (is_uptrend and 
                williams_r[i] < -80 and 
                volume_spike[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator downtrend, Williams %R overbought (> -20), volume spike, below 1-week EMA50
            elif (is_downtrend and 
                  williams_r[i] > -20 and 
                  volume_spike[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns ranging or Williams %R overbought (> -20)
            if (not is_uptrend or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns ranging or Williams %R oversold (< -80)
            if (not is_downtrend or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals