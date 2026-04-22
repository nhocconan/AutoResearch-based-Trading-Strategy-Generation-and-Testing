#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and strength.
# In trending markets (price outside Alligator mouth): trade in direction of alignment.
# In ranging markets (price inside Alligator mouth): no trades to avoid whipsaw.
# Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation requires current volume > 1.5x 20-period average.
# Designed to work in both bull and bear markets by avoiding ranging conditions.
# Targets 15-30 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (13, 8, 5 periods)
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = np.roll(jaw_raw.values, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = np.roll(teeth_raw.values, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = np.roll(lips_raw.values, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Alligator components and EMA50 to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Check for Alligator alignment (all three lines in order)
            # Bullish alignment: Lips > Teeth > Jaw
            # Bearish alignment: Lips < Teeth < Jaw
            bullish_aligned = lips_val > teeth_val > jaw_val
            bearish_aligned = lips_val < teeth_val < jaw_val
            
            # Only trade when Alligator is aligned (trending market)
            if bullish_aligned and price > lips_val and vol_spike and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            elif bearish_aligned and price < lips_val and vol_spike and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Teeth or Alligator loses alignment
                if price < teeth_val or not (lips_val > teeth_val > jaw_val):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Teeth or Alligator loses alignment
                if price > teeth_val or not (lips_val < teeth_val < jaw_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0