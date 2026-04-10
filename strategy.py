#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and chop regime filter
# - Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) + 1d volume > 1.5x 20-period volume SMA + Choppiness Index > 61.8 (ranging market)
# - Short when Lips < Teeth < Jaw (bearish alignment) + 1d volume > 1.5x 20-period volume SMA + Choppiness Index > 61.8
# - Exit: Opposite Alligator alignment (Lips crosses Teeth)
# - Position sizing: 0.25 discrete level
# - Alligator identifies trend initiation, volume confirms participation, chop filter ensures ranging conditions where Alligator works best
# - Works in bull/bear: Alligator catches new trends, chop filter prevents whipsaws in strong trends
# - 12h timeframe targets 30-60 trades/year with strict entry conditions

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted by 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted by 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA shifted by 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Calculate Choppiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    # Sum of TR over 14 periods
    sum_tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = 100 * np.log10(sum_tr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop = np.where((highest_high14 - lowest_low14) == 0, 100, chop)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA
        vol_confirm = volume_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: Choppiness Index > 61.8 indicates ranging market
        ranging_market = chop[i] > 61.8
        
        # Alligator alignment signals
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry conditions: Alligator alignment with volume and chop confirmation
        long_entry = bullish_alignment and vol_confirm and ranging_market
        short_entry = bearish_alignment and vol_confirm and ranging_market
        
        # Exit conditions: Opposite Alligator alignment (Lips crosses Teeth)
        long_exit = lips[i] < teeth[i]  # Exit long when Lips crosses below Teeth
        short_exit = lips[i] > teeth[i]  # Exit short when Lips crosses above Teeth
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals