#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike + 1w Trend Filter
Hypothesis: Williams Alligator identifies trend phases; aligned with 1w EMA55 trend and volume spikes
to capture strong momentum moves. Works in bull/bear by following major trends with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA55 for trend filter
    ema55_1w = pd.Series(df_1w['close'].values).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema55_1w_aligned = align_htf_to_ltf(prices, df_1w, ema55_1w)
    
    # Williams Alligator: SMA(13,8), SMA(8,5), SMA(5,3) on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema55_1w_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        trend = ema55_1w_aligned[i]
        vol_ok = vol_spike[i]
        # Alligator alignment: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Enter long on bullish alignment + volume spike + above weekly trend
            if bullish_align and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish alignment + volume spike + below weekly trend
            elif bearish_align and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on bearish alignment or below weekly trend
            if bearish_align or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish alignment or above weekly trend
            if bullish_align or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Volume_Spike_1wTrend"
timeframe = "1d"
leverage = 1.0