#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend presence when lines are aligned
# Jaw (13) > Teeth (8) > Lips (5) = bullish alignment
# Jaw (13) < Teeth (8) < Lips (5) = bearish alignment
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume spike (1.8x 20-period average) confirms institutional participation
# Designed for 6h timeframe to capture medium-term swings in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_Williams_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator components from 1d data
    # Alligator uses smoothed median prices (typical price = (high+low+close)/3)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Jaw: 13-period smoothed, shifted 8 bars forward
    jaw_raw = pd.Series(typical_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to forward shift
    
    # Teeth: 8-period smoothed, shifted 5 bars forward
    teeth_raw = pd.Series(typical_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to forward shift
    
    # Lips: 5-period smoothed, shifted 3 bars forward
    lips_raw = pd.Series(typical_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to forward shift
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate volume spike (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculation and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Jaw > Teeth > Lips
            bullish_alignment = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            # Bearish Alligator alignment: Jaw < Teeth < Lips
            bearish_alignment = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
            
            # Long: Bullish alignment + price > 1d EMA50 + volume spike
            if bullish_alignment and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < 1d EMA50 + volume spike
            elif bearish_alignment and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment (trend reversal)
            bearish_alignment = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
            if bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment (trend reversal)
            bullish_alignment = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            if bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals