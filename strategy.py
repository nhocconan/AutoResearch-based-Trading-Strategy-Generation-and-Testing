#!/usr/bin/env python3
"""
6h_ThreeSMA_Turn_Cross_v1
Hypothesis: Trend changes are signaled by crossovers of short, medium, and long SMAs.
The 6h timeframe captures medium-term momentum, and the 12h SMA acts as a trend filter to avoid whipsaws.
Entry occurs when SMA10 crosses SMA20 in the direction of the 12h SMA50 trend, with volume confirmation.
Exit occurs on the opposite SMA crossover. This aims to catch sustained moves while avoiding sideways chop.
Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
Target: 15-35 trades per year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h SMA50 for trend filter ===
    close_12h = df_12h['close'].values
    sma_50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    sma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    # === 6h SMA10 and SMA20 ===
    close = prices['close'].values
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(sma_10[i]) or np.isnan(sma_20[i]) or 
            np.isnan(sma_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        sma10_now = sma_10[i]
        sma20_now = sma_20[i]
        sma10_prev = sma_10[i-1]
        sma20_prev = sma_20[i-1]
        trend_12h = sma_50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Bullish crossover: SMA10 crosses above SMA20
        bullish_cross = (sma10_prev <= sma20_prev) and (sma10_now > sma20_now)
        # Bearish crossover: SMA10 crosses below SMA20
        bearish_cross = (sma10_prev >= sma20_prev) and (sma10_now < sma20_now)
        
        if position == 0:
            # Long: bullish crossover + price above 12h trend + volume
            if bullish_cross and (close[i] > trend_12h) and (vol_ratio_val > 1.2):
                signals[i] = 0.25
                position = 1
            # Short: bearish crossover + price below 12h trend + volume
            elif bearish_cross and (close[i] < trend_12h) and (vol_ratio_val > 1.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite crossover
            if position == 1 and bearish_cross:
                signals[i] = 0.0
                position = 0
            elif position == -1 and bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ThreeSMA_Turn_Cross_v1"
timeframe = "6h"
leverage = 1.0