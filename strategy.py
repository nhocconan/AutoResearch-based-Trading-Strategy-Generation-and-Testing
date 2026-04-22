#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) that act as dynamic support/resistance.
# In trending markets, the lines diverge (Green > Red > Blue for uptrend, reverse for downtrend).
# Combining with 1d EMA50 ensures alignment with higher timeframe trend, while volume spikes confirm momentum.
# Designed for low trade frequency (~20-40/year) to minimize fee decay. Works in both bull and bear markets
# by requiring trend alignment and avoiding choppy conditions where Alligator lines intertwine.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (waits for 1d bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 6h timeframe
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator alignment: Jaw > Teeth > Lips for uptrend, reverse for downtrend
        # Add small epsilon to avoid equality issues
        epsilon = 1e-8
        jaw_gt_teeth = jaw_val > teeth_val + epsilon
        teeth_gt_lips = teeth_val > lips_val + epsilon
        jaw_lt_teeth = jaw_val < teeth_val - epsilon
        teeth_lt_lips = teeth_val < lips_val - epsilon
        
        if position == 0:
            # Long conditions: Alligator aligned up + uptrend + volume spike
            if jaw_gt_teeth and teeth_gt_lips and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down + downtrend + volume spike
            elif jaw_lt_teeth and teeth_lt_lips and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Alligator alignment breaks down or trend breaks
                if not (jaw_gt_teeth and teeth_gt_lips) or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Alligator alignment breaks up or trend breaks
                if not (jaw_lt_teeth and teeth_lt_lips) or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0