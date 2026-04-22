#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# The Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips).
# When the averages are intertwined, the market is sleeping (range). When they diverge,
# the market is waking up (trend). We go long when Lips > Teeth > Jaw (bullish alignment)
# and short when Lips < Teeth < Jaw (bearish alignment), only when aligned with 1d EMA50 trend
# and confirmed by volume spike (>1.5x 20-period average). Designed for low trade frequency
# (~15-25/year on 12h) to minimize fee drift while capturing sustained trends in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    close = prices['close'].values
    
    # Calculate SMMA using EMA as approximation (close enough for our purposes)
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    
    # Apply the shifts (forward shift = lookback in calculation)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Fill NaN values from rolling
    jaw_shifted[:8] = jaw[:8] if len(jaw) > 8 else np.nan
    teeth_shifted[:5] = teeth[:5] if len(teeth) > 5 else np.nan
    lips_shifted[:3] = lips[:3] if len(lips) > 3 else np.nan
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator alignment conditions
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when alignment breaks or trend fails
                if not bullish_alignment or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when alignment breaks or trend fails
                if not bearish_alignment or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0