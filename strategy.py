#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume spike confirmation
# Williams Alligator identifies trending vs ranging markets via smoothed medians
# Long when Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA50 + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA50 + volume spike
# Uses tight entry conditions to target 12-30 trades/year, minimizing fee drag on 6h timeframe
# Works in both bull and bear markets due to 1d trend filter + volume confirmation avoiding whipsaws

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Williams Alligator on 6h timeframe (using median price = (high+low+close)/3)
    median_price = (high + low + close) / 3.0
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA of volume
        if i >= 20:
            vol_ema_20 = pd.Series(volume[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            volume_spike = volume[i] > (2.0 * vol_ema_20)
        else:
            volume_spike = False
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth OR Teeth < Jaw) OR price < 1d EMA50
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth OR Teeth > Jaw) OR price > 1d EMA50
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals