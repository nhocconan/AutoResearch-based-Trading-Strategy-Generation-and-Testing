#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator system with 1-day ATR filter
# Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d ATR < 1d ATR(50) (low volatility)
# Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d ATR < 1d ATR(50)
# Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order)
# Uses Alligator for trend detection, ATR for regime filter to avoid choppy markets
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)
name = "12h_Alligator_ATRRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Alligator and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: SMAs of median price (HL/2) with specific periods
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - using SMA as approximation
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1-day ATR for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align all 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        # Volatility regime: low volatility (ATR14 < ATR50)
        low_vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: bullish Alligator alignment + low volatility
            if bullish_alignment and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + low volatility
            elif bearish_alignment and low_vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bullish alignment breaks OR volatility increases
            if not bullish_alignment or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bearish alignment breaks OR volatility increases
            if not bearish_alignment or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals