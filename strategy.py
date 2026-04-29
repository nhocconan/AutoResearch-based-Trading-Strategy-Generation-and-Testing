#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combination with volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 1d timeframe for trend direction and alignment
# Elder Ray (Bull Power/Bear Power) from 12h for entry timing with EMA13
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Designed for 12h timeframe to capture medium-term swings with controlled frequency (target: 50-150 total trades)
# Works in both bull and bear markets: Alligator identifies trend, Elder Ray measures power, volume confirms validity

name = "12h_WilliamsAlligator_ElderRay_Volume_Combo"
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
    
    # Get 1d data for Williams Alligator (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator components from 1d
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Alligator Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = np.roll(jaw_1d, 8)
    jaw_1d[:8] = np.nan
    
    # Alligator Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(teeth_1d, 5)
    teeth_1d[:5] = np.nan
    
    # Alligator Lips (Green): 5-period SMMA, shifted 3 bars
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.roll(lips_1d, 3)
    lips_1d[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Get 12h data for Elder Ray and other calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Elder Ray components (Bull Power/Bear Power) using EMA13
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13_12h  # Bull Power = High - EMA13
    bear_power = low_12h - ema13_12h   # Bear Power = Low - EMA13
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 13)  # Alligator, volume MA, and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator reverses (Lips cross below Teeth) OR Bear Power becomes strongly negative
            if curr_lips < curr_teeth or curr_bear_power < -0.5 * np.std(bull_power[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips cross above Teeth) OR Bull Power becomes strongly positive
            if curr_lips > curr_teeth or curr_bull_power > 0.5 * np.std(bear_power[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Alligator alignment check: proper ordering for trend
            # Uptrend: Lips > Teeth > Jaw (Green > Red > Blue)
            # Downtrend: Lips < Teeth < Jaw (Green < Red < Blue)
            is_uptrend = curr_lips > curr_teeth > curr_jaw
            is_downtrend = curr_lips < curr_teeth < curr_jaw
            
            # Long entry: Uptrend + Bull Power positive + volume confirmation
            if vol_confirm and is_uptrend and curr_bull_power > 0:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Downtrend + Bear Power negative + volume confirmation
            elif vol_confirm and is_downtrend and curr_bear_power < 0:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals