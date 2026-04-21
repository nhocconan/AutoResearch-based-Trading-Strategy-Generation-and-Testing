#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Volume Spike + Price Channel (Donchian 10)
# Long when Jaw < Teeth < Lips (bullish alignment) + price > Donchian high(10) + 1d volume > 1.5x 20-day avg
# Short when Jaw > Teeth > Lips (bearish alignment) + price < Donchian low(10) + 1d volume > 1.5x 20-day avg
# Exit when alignment breaks or price crosses middle (Teeth)
# Williams Alligator identifies trend phases; volume confirms conviction; Donchian provides entry/exit levels
# Works in trending markets (both bull and bear) with volume filtering to avoid false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator lines (13,8,5 SMAs shifted by 8,5,3)
    close_1d = df_1d['close'].values
    sma13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    sma8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift as per Alligator: Jaw (13) shifted 8, Teeth (8) shifted 5, Lips (5) shifted 3
    jaw = np.roll(sma13, 8)
    teeth = np.roll(sma8, 5)
    lips = np.roll(sma5, 3)
    # Set invalid values for shifted periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels (10-period) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donch_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donch_mid = (donch_high + donch_low) / 2  # Middle line for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        volume = df_1d['volume'].iloc[i // 4] if i >= 4 else df_1d['volume'].iloc[0]  # 4 bars per day (24h/6h)
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma if i >= 4 else df_1d['volume'].iloc[0] > 1.5 * vol_ma
        
        # Alligator alignment
        bullish_align = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_align = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish alignment + price > Donchian high + volume confirmation
            if bullish_align and price > donch_high[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < Donchian low + volume confirmation
            elif bearish_align and price < donch_low[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if alignment breaks or price crosses below Donchian mid
                if not bullish_align or price < donch_mid[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if alignment breaks or price crosses above Donchian mid
                if not bearish_align or price > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dVolumeSpike_Donchian10"
timeframe = "6h"
leverage = 1.0