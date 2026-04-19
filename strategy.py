#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw, Teeth, Lips) + volume spike + ADX trend filter.
# The Alligator uses SMAs to detect trend alignment: Jaw (13-period), Teeth (8-period), Lips (5-period).
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw.
# Combined with volume confirmation (2x 20-period average) and ADX > 25 for trend strength.
# Designed for 12h timeframe to capture strong trends with low frequency, suitable for both bull and bear markets.
# Entry: Long when Lips > Teeth > Jaw and volume spike and ADX > 25; Short when Lips < Teeth < Jaw and volume spike and ADX > 25.
# Exit: Opposite Alligator alignment (Lips < Teeth for long exit, Lips > Teeth for short exit).
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "12h_WilliamsAlligator_ADX_Volume"
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
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw: 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # Teeth: 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Lips: 5-period
    
    # ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend alignment) with volume and trend strength
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume_spike[i] and adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend alignment) with volume and trend strength
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume_spike[i] and adx[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if uptrend breaks (Lips < Teeth)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if downtrend breaks (Lips > Teeth)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals