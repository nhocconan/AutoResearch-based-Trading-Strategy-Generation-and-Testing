#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator trend following with 1d trend filter
# Uses 1d EMA50 for trend filter and Williams Alligator (Jaw/Teeth/Lips) on 4h for entry
# Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs
# Long when Lips > Teeth > Jaw and price > 1d EMA50
# Short when Lips < Teeth < Jaw and price < 1d EMA50
# Designed for low trade frequency (15-40/year) to avoid fee drag in 4h timeframe
# Works in trending markets and avoids whipsaws in ranging markets via Alligator convergence/divergence
name = "4h_WilliamsAlligator_1dEMA50_Trend"
timeframe = "4h"
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
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h data
    # Jaw: 13-period SMMA, smoothed 8 periods ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean()
    
    # Teeth: 8-period SMMA, smoothed 5 periods ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean()
    
    # Lips: 5-period SMMA, smoothed 3 periods ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean()
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient data for Alligator components
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h[i]) or np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_values[i] < teeth_values[i] and teeth_values[i] < jaw_values[i]
        
        if position == 0:
            # Long: bullish alignment and price above 1d EMA50
            if bullish_alignment and close[i] > ema_50_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment and price below 1d EMA50
            elif bearish_alignment and close[i] < ema_50_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or price below 1d EMA50
            if bearish_alignment or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or price above 1d EMA50
            if bullish_alignment or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals