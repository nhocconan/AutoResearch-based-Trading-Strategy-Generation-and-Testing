#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x average.
# Exit when Alligator alignment breaks (jaws > teeth OR teeth > lips).
# Uses 1d volume filter to reduce noise and focus on institutional participation.
# Target: 15-35 trades/year to minimize fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (13,8,5 SMAs of median price)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Median price = (high + low) / 2
    median_price_12h = (high_12h + low_12h) / 2
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for Elder Ray and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Volume filter: 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align Elder Ray and volume filter to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish Alligator alignment: jaws < teeth < lips
        bullish_alligator = (jaw_12h_aligned[i] < teeth_12h_aligned[i] and 
                            teeth_12h_aligned[i] < lips_12h_aligned[i])
        # Bearish Alligator alignment: jaws > teeth > lips
        bearish_alligator = (jaw_12h_aligned[i] > teeth_12h_aligned[i] and 
                            teeth_12h_aligned[i] > lips_12h_aligned[i])
        
        # Long condition: bullish Alligator + bullish Elder Ray + volume confirmation
        if (bullish_alligator and 
            bull_power_aligned[i] > 0 and 
            volume_filter_aligned[i] > 0.5):  # True when aligned
            signals[i] = 0.25
            position = 1
        # Short condition: bearish Alligator + bearish Elder Ray + volume confirmation
        elif (bearish_alligator and 
              bear_power_aligned[i] < 0 and 
              volume_filter_aligned[i] > 0.5):
            signals[i] = -0.25
            position = -1
        # Exit conditions: Alligator alignment breaks
        elif position == 1 and not bullish_alligator:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not bearish_alligator:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeFilter"
timeframe = "12h"
leverage = 1.0