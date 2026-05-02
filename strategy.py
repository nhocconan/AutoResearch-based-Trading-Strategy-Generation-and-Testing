#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d Elder Ray trend filter and volume confirmation
# Uses 12h primary timeframe for signal generation with Williams Alligator (jaw/teeth/lips)
# Bullish: lips > teeth > jaw + price > lips; Bearish: lips < teeth < jaw + price < lips
# 1d Elder Ray trend filter: Bull Power > 0 and Bear Power < 0 for strong trend bias
# Volume confirmation (1.8x 30-period average) filters for strong participation to reduce false breakouts
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trending vs ranging markets; Elder Ray confirms trend strength
# Works in both bull and bear markets by only trading in strong trend direction with volume confirmation

name = "12h_WilliamsAlligator_1dElderRay_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1d['high'].values - ema13_1d)
    bear_power = (df_1d['low'].values - ema13_1d)
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'] + df_12h['low']) / 2
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components (already on 12h, but ensure alignment for safety)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: lips > teeth > jaw
            bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Bearish Alligator: lips < teeth < jaw
            bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            # Long: Bullish Alligator + price > lips + Elder Ray bullish + volume spike
            if (bullish_alligator and close[i] > lips_aligned[i] and 
                bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + price < lips + Elder Ray bearish + volume spike
            elif (bearish_alligator and close[i] < lips_aligned[i] and 
                  bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price < lips OR Elder Ray turns bearish
            bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            if bearish_alligator or close[i] < lips_aligned[i] or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price > lips OR Elder Ray turns bullish
            bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            if bullish_alligator or close[i] > lips_aligned[i] or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals