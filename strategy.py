#!/usr/bin/env python3
"""
#100810 - 4h_Price_Action_Mean_Reversion_with_Volume_Confirmation
Hypothesis: Mean reversion strategy using price action at key levels with volume confirmation.
Buys when price rejects support with volume, sells when price rejects resistance with volume.
Works in both bull (buying dips) and bear (selling rallies) markets by fading extremes.
Uses 4h timeframe with 1d support/resistance levels and volume spike confirmation.
Target: 20-40 trades/year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d high/low for support/resistance (using previous day to avoid look-ahead)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate key levels: previous day high/low and midpoint
    resistance = prev_high
    support = prev_low
    midpoint = (prev_high + prev_low) / 2
    
    # Align to 4h timeframe
    resistance_4h = align_htf_to_ltf(prices, df_1d, resistance)
    support_4h = align_htf_to_ltf(prices, df_1d, support)
    midpoint_4h = align_htf_to_ltf(prices, df_1d, midpoint)
    
    # Volume confirmation: volume > 1.8x 24-period average (6 hours)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Price action signals: rejection of levels with volume
    signals = np.zeros(n)
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(resistance_4h[i]) or np.isnan(support_4h[i]) or 
            np.isnan(midpoint_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Buy signal: price near support AND volume spike AND rejection (close > open)
        near_support = close[i] <= support_4h[i] * 1.005  # Within 0.5% of support
        rejection_up = close[i] > prices['open'].iloc[i]  # Bullish candle
        
        if near_support and volume_spike[i] and rejection_up:
            signals[i] = 0.25
            
        # Sell signal: price near resistance AND volume spike AND rejection (close < open)
        elif close[i] >= resistance_4h[i] * 0.995:  # Within 0.5% of resistance
            near_resistance = True
            rejection_down = close[i] < prices['open'].iloc[i]  # Bearish candle
            
            if near_resistance and volume_spike[i] and rejection_down:
                signals[i] = -0.25
                
        # Exit when price reaches midpoint (mean reversion target)
        elif i > start_idx:
            if signals[i-1] > 0 and close[i] >= midpoint_4h[i]:
                signals[i] = 0.0
            elif signals[i-1] < 0 and close[i] <= midpoint_4h[i]:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]  # Hold position
    
    return signals

name = "4h_Price_Action_Mean_Reversion_with_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0