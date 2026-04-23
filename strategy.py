#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA filter and volume confirmation.
Long when price > Alligator Jaw AND Jaw > Teeth AND Teeth > Lips (bullish alignment) AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips (bearish alignment) AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Alligator alignment breaks (Jaw-Teeth-Lips not monotonic).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams Alligator (SMAs of median price) captures trends with built-in smoothing. 1d EMA50 provides HTF trend filter. Volume confirmation ensures breakout validity.
Designed to work in both bull and bear markets by requiring strict alignment and HTF trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator on primary timeframe (12h)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Alligator components: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    # Using SMA as approximation for SMMA with proper alignment
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align HTF indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 13+8, 8+5, 5+3)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        # Check Alligator alignment
        bullish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        bearish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        
        if position == 0:
            # Long: bullish alignment AND close > 1d EMA50 AND volume spike
            if (bullish_alignment and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND close < 1d EMA50 AND volume spike
            elif (bearish_alignment and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator alignment breaks
            if position == 1 and not bullish_alignment:
                exit_signal = True
            elif position == -1 and not bearish_alignment:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0