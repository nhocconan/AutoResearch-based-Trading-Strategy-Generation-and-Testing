#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) representing market equilibrium.
# When the Alligator is "sleeping" (lines intertwined) -> ranging market -> fade extremes
# When the Alligator "awakens" (lines diverge) -> trending market -> trade breakout
# Long: Lips > Teeth > Jaw (bullish alignment) + price above 1w EMA50 + volume spike
# Short: Lips < Teeth < Jaw (bearish alignment) + price below 1w EMA50 + volume spike
# Uses 6h timeframe for lower frequency (target: 12-37 trades/year) to minimize fee drag
# Alligator acts as trend/filter; volume spike confirms momentum; 1w EMA50 ensures higher timeframe alignment

name = "6h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: three SMAs (Jaw=13, Teeth=8, Lips=5)
    # Jaw (Blue) - 13-period SMMA shifted 8 bars ahead
    # Teeth (Red) - 8-period SMMA shifted 5 bars ahead  
    # Lips (Green) - 5-period SMMA shifted 3 bars ahead
    # Note: Using regular SMA with min_periods as SMMA approximation
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 13, 8, 5, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment, volume spike, uptrend
            if bullish_alignment and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment, volume spike, downtrend
            elif bearish_alignment and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or trend reversal
            if bearish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or trend reversal
            if bullish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals