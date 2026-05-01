#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator strategy with 1d trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
# Enter long when Lips cross above Teeth and Jaw in uptrend with volume spike.
# Enter short when Lips cross below Teeth and Jaw in downtrend with volume spike.
# Uses 1d EMA50 for medium-term trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by trading with the 1d trend.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h data: SMAs with specific periods and offsets
    # Jaw: 13-period SMMA, offset 8 bars
    # Teeth: 8-period SMMA, offset 5 bars  
    # Lips: 5-period SMMA, offset 3 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (need enough for Jaw shift)
    start_idx = 20  # Need sufficient history for Alligator components
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (green alignment)
        bullish_align = lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bearish_align = lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment, volume spike, uptrend
            if bullish_align and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, volume spike, downtrend
            elif bearish_align and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or trend reversal
            if bearish_align or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or trend reversal
            if bullish_align or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals