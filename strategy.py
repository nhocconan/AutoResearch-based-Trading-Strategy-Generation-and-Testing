#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Long when Jaw < Teeth < Lips (bullish alignment), volume > 1.5x 20-day avg, CHOP > 61.8 (range)
# Short when Jaw > Teeth > Lips (bearish alignment), volume > 1.5x 20-day avg, CHOP > 61.8 (range)
# Exit when alignment breaks or CHOP < 38.2 (trend)
# Williams Alligator identifies trend phases via SMAs: Jaw=13, Teeth=8, Lips=5 (all shifted)
# Chop regime filter ensures we only trade in ranging markets where mean reversion works
# Volume spike confirms conviction
# Target: 15-25 trades/year by requiring triple confluence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components (all SMAs with shift)
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chopiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_14 = pd.Series(np.maximum(np.maximum(high_1d - low_1d, 
                                             np.abs(high_1d - np.roll(close_1d, 1))), 
                                  np.abs(low_1d - np.roll(close_1d, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log(14) / (highest_high - lowest_low))
    
    # Align all 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h price for entry
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Chop warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        vol_idx = i // 2  # 2 bars per day (24h/12h)
        if vol_idx >= len(df_1d):
            vol_idx = len(df_1d) - 1
        volume = df_1d['volume'].iloc[vol_idx] if vol_idx >= 0 else df_1d['volume'].iloc[0]
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume > 1.5 * vol_ma
        
        # Alligator alignment conditions
        bullish_align = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_align = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        # Chop regime: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Long: Bullish alignment + volume confirmation + ranging market
            if bullish_align and volume_confirm and in_range:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume confirmation + ranging market
            elif bearish_align and volume_confirm and in_range:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bullish alignment breaks OR market starts trending
                if not bullish_align or chop_val < 38.2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bearish alignment breaks OR market starts trending
                if not bearish_align or chop_val < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0