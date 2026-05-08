#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-week trend filter and volume confirmation
# The Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# We go long when Lips > Teeth > Jaw (bullish alignment) with price above Teeth,
# weekly close above weekly EMA(34), and volume spike.
# We go short when Lips < Teeth < Jaw (bearish alignment) with price below Teeth,
# weekly close below weekly EMA(34), and volume spike.
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (blue line): 13-period SMMA, smoothed with 8-period shift
    # Teeth (red line): 8-period SMMA, smoothed with 5-period shift  
    # Lips (green line): 5-period SMMA, smoothed with 3-period shift
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.convolve(arr, np.ones(period)/period, mode='valid')
        result[period-1:] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: bullish alignment (Lips > Teeth > Jaw) + price above Teeth +
            # weekly uptrend + volume spike
            if (lips_val > teeth_val > jaw_val and 
                close[i] > teeth_val and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (Lips < Teeth < Jaw) + price below Teeth +
            # weekly downtrend + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  close[i] < teeth_val and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price below Teeth OR weekly trend turns down
            if (lips_val < teeth_val or close[i] < teeth_val or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price above Teeth OR weekly trend turns up
            if (lips_val > teeth_val or close[i] > teeth_val or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals