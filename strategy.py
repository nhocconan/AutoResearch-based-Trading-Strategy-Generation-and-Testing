#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA Trend Filter + Volume Spike
# Williams Alligator (Jaw, Teeth, Lips) identifies trend direction and momentum.
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 12h EMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 12h EMA50.
# Volume confirmation requires > 1.8x 20-bar median volume to filter low-quality breakouts.
# Designed to capture strong trends in both bull and bear markets while avoiding whipsaws.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Williams Alligator on 6h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Volume confirmation: current > 1.8x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for Alligator
        # Skip if any required data is NaN
        if (np.isnan(lips.iloc[i]) or np.isnan(teeth.iloc[i]) or 
            np.isnan(jaw.iloc[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        lips_val = lips.iloc[i]
        teeth_val = teeth.iloc[i]
        jaw_val = jaw.iloc[i]
        close_val = close[i]
        ema_val = ema_12h_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        # Long: Bullish Alligator, price above 12h EMA50, volume spike
        if (bullish and 
            close_val > ema_val and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bearish Alligator, price below 12h EMA50, volume spike
        elif (bearish and 
              close_val < ema_val and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Alligator alignment breaks or price crosses 12h EMA in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish or close_val < ema_val)) or
               (signals[i-1] == -0.25 and (not bearish or close_val > ema_val)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0