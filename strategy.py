#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) confluence
# Williams Alligator (jaw/teeth/lips) identifies trend direction and strength on 6h.
# 1d Elder Ray measures bull/bear power relative to 13-period EMA to filter counter-trend trades.
# Long when: Alligator bullish (lips > teeth > jaw) AND 1d Bull Power > 0
# Short when: Alligator bearish (lips < teeth < jaw) AND 1d Bear Power < 0
# Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsAlligator_1dElderRay_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h Williams Alligator (SMMA = smoothed moving average)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    # Jaw: SMMA(13, 8) - 13-period smoothed moving average, 8 bars ahead
    jaw_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMMA(8, 5) - 8-period smoothed moving average, 5 bars ahead
    teeth_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMMA(5, 3) - 5-period smoothed moving average, 3 bars ahead
    lips_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 6h Alligator to 6h timeframe (no additional delay needed as it's based on current bar)
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    
    # 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # 1d EMA(13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe (no additional delay as it's contemporaneous)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for longest indicator
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions
        alligator_bullish = lips_6h_aligned[i] > teeth_6h_aligned[i] > jaw_6h_aligned[i]
        alligator_bearish = lips_6h_aligned[i] < teeth_6h_aligned[i] < jaw_6h_aligned[i]
        
        # 1d Elder Ray conditions
        bull_power_pos = bull_power_1d_aligned[i] > 0
        bear_power_neg = bear_power_1d_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Bull Power positive
            if alligator_bullish and bull_power_pos:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power negative
            elif alligator_bearish and bear_power_neg:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator bearish OR Bear Power negative (trend weakening)
            if alligator_bearish or bear_power_neg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator bullish OR Bull Power positive (trend weakening)
            if alligator_bullish or bull_power_pos:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals