#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw, Teeth, Lips) that indicate market phases:
# - When lines are intertwined: market is sleeping (range) - avoid trading
# - When lines diverge upward: bullish trend forming
# - When lines diverge downward: bearish trend forming
# Strategy: Enter when Alligator shows clear trend alignment (jaws < teeth < lips for up, reverse for down)
# Filter with 1d EMA34 trend and volume spike (>1.5x 20-period average)
# Designed for ~20-30 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (all SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # Using SMA as proxy for SMMA (close enough for this application)
    
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # Apply the forward shifts (using NaN padding)
    jaw = np.full_like(close, np.nan)
    teeth = np.full_like(close, np.nan)
    lips = np.full_like(close, np.nan)
    
    jaw[8:] = jaw_raw[:-8].values if len(jaw_raw) > 8 else np.nan
    teeth[5:] = teeth_raw[:-5].values if len(teeth_raw) > 5 else np.nan
    lips[3:] = lips_raw[:-3].values if len(lips_raw) > 3 else np.nan
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals:
        # Bullish alignment: Lips > Teeth > Jaw (green alignment)
        # Bearish alignment: Lips < Teeth < Jaw (red alignment)
        # Avoid trading when intertwined (no clear trend)
        
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if bullish_alignment and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
            # Strong bullish alignment with uptrend filter and volume
            signals[i] = 0.25
            position = 1
        elif bearish_alignment and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
            # Strong bearish alignment with downtrend filter and volume
            signals[i] = -0.25
            position = -1
        else:
            # Hold position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0