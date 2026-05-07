#!/usr/bin/env python3
name = "4h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Williams Alligator components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate SMoothed Moving Average (SMMA) - using EMA as approximation
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_1d = pd.Series(median_price_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw_1d = np.roll(jaw_1d, 8)  # Shift forward 8 bars
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth_1d = np.roll(teeth_1d, 5)  # Shift forward 5 bars
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips_1d = np.roll(lips_1d, 3)  # Shift forward 3 bars
    
    # Align to 4h timeframe with additional delay for confirmation
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d, additional_delay_bars=0)
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for Alligator components
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw (alligator mouth opening up)
            bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Bearish alignment: Jaw > Teeth > Lips (alligator mouth opening down)
            bearish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            if bullish and vol_condition:
                signals[i] = 0.25
                position = 1
            elif bearish and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: alligator lines converge (Lips <= Teeth) or volume drops
            if lips_aligned[i] <= teeth_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: alligator lines converge (Teeth <= Lips) or volume drops
            if teeth_aligned[i] <= lips_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Williams Alligator with 1d trend confirmation and volume spike
# - Williams Alligator identifies trending vs ranging markets via jaw/teeth/lips alignment
# - Bullish: Lips > Teeth > Jaw (mouth opening up) - enter long
# - Bearish: Jaw > Teeth > Lips (mouth opening down) - enter short
# - Requires 1.5x volume spike to confirm institutional participation
# - Exit when alligator lines converge (trend weakening) or volume drops
# - Works in both bull (buy in bullish alignment) and bear (sell in bearish alignment)
# - Position size 0.25 targets ~20-30 trades/year, avoiding fee drag
# - Uses 1d timeframe for Alligator to avoid whipsaws, aligned to 4h for execution