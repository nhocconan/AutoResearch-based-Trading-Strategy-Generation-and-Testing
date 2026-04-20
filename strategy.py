#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Volume Spike + Chop Regime
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) on 4h for trend detection
# - Long when Lips > Teeth > Jaw and price > Lips (bullish alignment)
# - Short when Lips < Teeth < Jaw and price < Lips (bearish alignment)
# - 1d volume spike (current volume > 1.5 * 20-day average) confirms institutional participation
# - Chop regime filter: only trade when Chop(14) < 38.2 (trending market)
# - Designed for 4h timeframe with trend-following in strong markets, avoiding range-bound periods
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike indicator
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate 1d Chop index (14-period)
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean()
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / 
                          np.log10(max_high_14 - min_low_14)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate Williams Alligator on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw_raw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth_raw = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips_raw = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Alligator warmup
        # Skip if NaN in indicators
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or 
            np.isnan(lips_values[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        jaw_val = jaw_values[i]
        teeth_val = teeth_values[i]
        lips_val = lips_values[i]
        vol_spike = volume_spike_aligned[i] > 0.5
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long entry: Bullish alignment + volume spike + trending market
            if (lips_val > teeth_val > jaw_val and 
                price > lips_val and 
                vol_spike and 
                chop_val < 38.2):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + volume spike + trending market
            elif (lips_val < teeth_val < jaw_val and 
                  price < lips_val and 
                  vol_spike and 
                  chop_val < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or loss of volume spike or choppy market
            if (lips_val < teeth_val or 
                not vol_spike or 
                chop_val > 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or loss of volume spike or choppy market
            if (lips_val > teeth_val or 
                not vol_spike or 
                chop_val > 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0