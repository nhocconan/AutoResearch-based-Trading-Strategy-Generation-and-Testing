#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1-day volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trend direction
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
# Entry confirmed by 1-day volume spike (>1.5x 20-day average)
# Exit when Alligator lines re-cross or volume drops below average
# Designed to capture trends while avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Williams Alligator components on 12h timeframe
    close = prices['close'].values
    
    # Jaw: 13-period SMMA (smoothed moving average)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    # Teeth: 8-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
    # Lips: 5-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current day volume > 1.5x 20-day average
        # Get current day's volume (need to map 12h bar to day)
        day_idx = i // 2  # 2 twelve-hour bars per day
        if day_idx >= len(volume_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        current_volume = volume_1d[day_idx]
        vol_spike = current_volume > (1.5 * vol_ma_20_aligned[i*2]) if (i*2) < len(vol_ma_20_aligned) else False
        
        price = close[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator re-crosses against trend OR volume drops
            if lips[i] < teeth[i] or teeth[i] < jaw[i]:  # Bullish alignment broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator re-crosses against trend OR volume drops
            if lips[i] > teeth[i] or teeth[i] > jaw[i]:  # Bearish alignment broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike"
timeframe = "12h"
leverage = 1.0