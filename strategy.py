#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and entry points
# Combined with 1d EMA50 for higher timeframe trend confirmation and volume filter to reduce false signals
# Designed for 12h timeframe with target of 50-150 trades over 4 years (12-37/year)
# Works in bull/bear markets by requiring alignment between Alligator jaws and higher timeframe trend
name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (13-period SMMA shifted 8 bars forward)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    jaw_values = jaw.values
    
    # Teeth (8-period SMMA shifted 5 bars forward)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    teeth_values = teeth.values
    
    # Lips (5-period SMMA shifted 3 bars forward)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    lips_values = lips.values
    
    # Align Alligator components (already on 12h timeframe, no additional alignment needed)
    # But we still need to handle NaN values properly
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        # Bullish alignment: Lips > Teeth > Jaw (alligator opening upwards)
        bullish_alignment = lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i]
        # Bearish alignment: Lips < Teeth < Jaw (alligator opening downwards)
        bearish_alignment = lips_values[i] < teeth_values[i] and teeth_values[i] < jaw_values[i]
        
        # Price above/below Alligator mouth
        price_above_mouth = close[i] > lips_values[i]
        price_below_mouth = close[i] < lips_values[i]
        
        # Trend filter from 1d EMA50
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + price above lips + uptrend + volume confirmation
            if bullish_alignment and price_above_mouth and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + price below lips + downtrend + volume confirmation
            elif bearish_alignment and price_below_mouth and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Alligator alignment or price below lips or trend reversal
            if bearish_alignment or not price_above_mouth or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Alligator alignment or price above lips or trend reversal
            if bullish_alignment or not price_below_mouth or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals