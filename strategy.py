#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + volume confirmation + 12h EMA(50) trend filter
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > Jaw + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < Jaw + volume spike
# Uses 12h EMA(50) for stronger trend alignment to reduce whipsaw in choppy markets
# Designed for low trade frequency (19-50/year) to minimize fee drag. Works in both bull and bear markets.

name = "4h_WilliamsAlligator_Volume_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator components on 4h
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period smoothed median (SMMA)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    
    # Teeth: 8-period smoothed median (SMMA)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    
    # Lips: 5-period smoothed median (SMMA)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 60  # max(13,8,5 for Alligator + 50 for 12h EMA + 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: Bullish Alligator + price > Jaw + above 12h EMA(50) + volume spike
            if (bullish_alligator and close[i] > jaw[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator + price < Jaw + below 12h EMA(50) + volume spike
            elif (bearish_alligator and close[i] < jaw[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR price below Jaw OR below 12h EMA(50)
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alligator or close[i] < jaw[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR price above Jaw OR above 12h EMA(50)
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alligator or close[i] > jaw[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals