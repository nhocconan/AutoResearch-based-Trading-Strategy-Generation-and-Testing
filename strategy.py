#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
# Trend: Alligator sleeping (SMA convergence) -> awakening (divergence) -> feeding (strong trend)
# Entry: Go long when Lips > Teeth > Jaw (bullish alignment) with volume spike and 1d EMA34 uptrend
# Entry: Go short when Lips < Teeth < Jaw (bearish alignment) with volume spike and 1d EMA34 downtrend
# Exit: Position reversal or loss of alignment
# Uses discrete sizing (0.25) to minimize fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator components (SMAs)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # Warmup for 1d EMA34, volume MA, and Alligator Jaw
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Alligator alignment conditions
        bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: loss of bullish alignment OR breaks 1d EMA34 trend
            if not bullish_alignment or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: loss of bearish alignment OR breaks 1d EMA34 trend
            if not bearish_alignment or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment AND above 1d EMA34 AND volume spike
            if bullish_alignment and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment AND below 1d EMA34 AND volume spike
            elif bearish_alignment and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals