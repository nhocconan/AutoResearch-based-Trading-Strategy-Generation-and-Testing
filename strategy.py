#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-week Trend and Volume Confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) and 1-week EMA50 is rising with volume spike.
Short when Alligator jaws > teeth > lips (bearish alignment) and 1-week EMA50 is falling with volume spike.
Exit when Alligator alignment breaks or 1-week EMA50 reverses.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-week trend.
"""

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
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    # Jaws: 13-period SMA, 8 bars ahead
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-week close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bullish alignment: jaws < teeth < lips
            bullish_align = jaws[i] < teeth[i] < lips[i]
            # Bearish alignment: jaws > teeth > lips
            bearish_align = jaws[i] > teeth[i] > lips[i]
            
            # Long: Bullish alignment, 1w EMA50 rising, volume spike
            if (bullish_align and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, 1w EMA50 falling, volume spike
            elif (bearish_align and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alignment breaks or 1w EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Bullish alignment breaks or 1w EMA50 turns down
                bullish_align = jaws[i] < teeth[i] < lips[i]
                if not bullish_align or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bearish alignment breaks or 1w EMA50 turns up
                bearish_align = jaws[i] > teeth[i] > lips[i]
                if not bearish_align or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0