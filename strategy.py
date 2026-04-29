#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to detect trend exhaustion
# In bear markets: price below alligator jaws with Teeth < Lips = short setup
# In bull markets: price above alligator jaws with Teeth > Lips = long setup
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume spike (>1.8x 20-period average) confirms institutional participation
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Alligator components: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator to daily timeframe (already aligned, but ensure no look-ahead)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for Alligator and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price above Jaw AND Teeth > Lips (aligned) AND above weekly EMA50
                if (curr_close > curr_jaw and curr_teeth > curr_lips and curr_close > curr_ema_50):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below Jaw AND Teeth < Lips (aligned) AND below weekly EMA50
                elif (curr_close < curr_jaw and curr_teeth < curr_lips and curr_close < curr_ema_50):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price closes below Jaw (trend exhaustion)
            if curr_close < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above Jaw (trend exhaustion)
            if curr_close > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals