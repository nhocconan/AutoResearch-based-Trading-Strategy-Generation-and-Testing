#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation.
# Williams Alligator uses SMAs of median price (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In trending markets (price above/below all lines with separation): trade in direction of alignment.
# In ranging markets (lines intertwined): avoid trading.
# 1w EMA filter ensures we only trade in alignment with higher timeframe trend.
# Volume confirmation (>1.3x average) reduces false signals.
# Designed for 12h timeframe to capture medium-term trends with low trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1w data (higher timeframe for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Williams Alligator on 12h ===
    # Median price = (high + low) / 2
    median_price = (high_12h + low_12h) / 2
    
    # Jaw (13-period SMMA of median, shifted 8 bars)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (8-period SMMA of median, shifted 5 bars)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (5-period SMMA of median, shifted 3 bars)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Align Alligator components to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips.values)
    
    # === 1w EMA(34) for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_1w = ema_34_1w_aligned[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below the lowest of Jaw/Teeth/Lips
            if price < min(jaw_val, teeth_val, lips_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above the highest of Jaw/Teeth/Lips
            if price > max(jaw_val, teeth_val, lips_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Alligator lines converge or price crosses below Teeth
            if (jaw_val <= teeth_val or teeth_val <= lips_val) or price < teeth_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Alligator lines converge or price crosses above Teeth
            if (jaw_val >= teeth_val or teeth_val >= lips_val) or price > teeth_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Check if Alligator lines are separated (trending market)
            lips_above_teeth = lips_val > teeth_val
            teeth_above_jaw = teeth_val > jaw_val
            
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_aligned = lips_above_teeth and teeth_above_jaw
            
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
            
            if bullish_aligned and price > ema_1w and vol_ratio > 1.3:
                # Long: price above 1w EMA, bullish Alligator, volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif bearish_aligned and price < ema_1w and vol_ratio > 1.3:
                # Short: price below 1w EMA, bearish Alligator, volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1wEMA_Volume_v1"
timeframe = "12h"
leverage = 1.0