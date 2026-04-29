#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation
# Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
# Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Alligator identifies trend initiation/continuation; EMA50 filters counter-trend noise; volume confirms conviction
# Works in bull via trend continuation, in bear via trend reversal signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Apply offsets: Jaw offset 8, Teeth offset 5, Lips offset 3
    jaw_offset = np.roll(jaw, 8)
    teeth_offset = np.roll(teeth, 5)
    lips_offset = np.roll(lips, 3)
    
    # Set NaN for invalid offset positions
    jaw_offset[:8] = np.nan
    teeth_offset[:5] = np.nan
    lips_offset[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5) + 8  # warmup for Alligator + max offset
    
    for i in range(start_idx, n):
        # Skip if any Alligator value is NaN
        if np.isnan(lips_offset[i]) or np.isnan(teeth_offset[i]) or np.isnan(jaw_offset[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Williams Alligator signals
        lips_val = lips_offset[i]
        teeth_val = teeth_offset[i]
        jaw_val = jaw_offset[i]
        
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Loss of bullish alignment OR price below 1d EMA50
            if not bullish_alignment or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Loss of bearish alignment OR price above 1d EMA50
            if not bearish_alignment or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish alignment AND price > 1d EMA50 AND volume spike
            if bullish_alignment and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment AND price < 1d EMA50 AND volume spike
            elif bearish_alignment and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals