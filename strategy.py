#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Trend Filter
# Uses Williams Alligator (Jaw, Teeth, Lips) on 6h to detect trends. Only take signals when
# 1d EMA50 shows aligned trend (price > EMA50 for long, price < EMA50 for short).
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3). 
# Long when Lips > Teeth > Jaw (bullish alignment). Short when Lips < Teeth < Jaw (bearish).
# Works in both bull and bear markets by following the trend. Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 6h
    # Jaw: SMA(13, 8) - median price smoothed with 8-period lag
    # Teeth: SMA(8, 5) - median price smoothed with 5-period lag  
    # Lips: SMA(5, 3) - median price smoothed with 3-period lag
    median_price = (high + low) / 2
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # 8-period lag
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # 5-period lag
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # 3-period lag
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw  
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long entry: bullish alignment + price above 1d EMA50
        if bullish and close[i] > ema50_1d_aligned[i] and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + price below 1d EMA50
        elif bearish and close[i] < ema50_1d_aligned[i] and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or price crosses 1d EMA50
        elif position == 1 and (not bullish or close[i] < ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish or close[i] > ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50"
timeframe = "6h"
leverage = 1.0