#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + 1w EMA Trend Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h to identify trend direction and alignment.
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 1w EMA200 (long-term uptrend).
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 1w EMA200 (long-term downtrend).
# Volume confirmation requires current volume > 2.0x 50-bar median volume to avoid false signals.
# Designed to work in both bull and bear markets by requiring long-term trend alignment.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA200 for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams Alligator on 12h
    df_12h = get_htf_data(prices, '12h')
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (Green): 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips.values)
    
    # Volume confirmation: current > 2.0x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Long: Bullish alignment, price above 1w EMA200, volume spike
        if (bullish and 
            close[i] > ema_200_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bearish alignment, price below 1w EMA200, volume spike
        elif (bearish and 
              close[i] < ema_200_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Alignment breaks or price crosses 1w EMA200 in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish or close[i] <= ema_200_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (not bearish or close[i] >= ema_200_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_1wEMA_Volume"
timeframe = "12h"
leverage = 1.0