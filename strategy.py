#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years on 12h.
# Williams Alligator is a trend-following system that works well in both bull and bear markets by identifying
# when the market is trending (aligned Alligator) vs ranging (intertwined Alligator).

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply the forward shifts (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (i < 8 or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        bullish_alignment = lips_val > teeth_val > jaw_val   # Lips > Teeth > Jaw
        bearish_alignment = lips_val < teeth_val < jaw_val   # Lips < Teeth < Jaw
        
        # Entry logic
        if position == 0:
            # Long: bullish Alligator alignment AND 1d uptrend AND volume spike
            if bullish_alignment and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND 1d downtrend AND volume spike
            elif bearish_alignment and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR 1d trend turns down
            if bearish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR 1d trend turns up
            if bullish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals