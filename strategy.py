#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + 12h EMA Trend Filter
Based on Bill Williams Alligator (Jaw/Teeth/Lips SMAs) to detect trend presence.
Long when Lips > Teeth > Jaw (bullish alignment) with volume spike.
Short when Lips < Teeth < Jaw (bearish alignment) with volume spike.
Uses 12h EMA34 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency with clear trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 12h EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend direction
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Williams Alligator components (13, 8, 5 period SMAs with future shift)
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment signals
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        price = close[i]
        above_12h_ema = price > ema_34_12h_aligned[i]
        below_12h_ema = price < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: bullish alignment, price above 12h EMA, volume spike
            if (bullish_alignment and above_12h_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below 12h EMA, volume spike
            elif (bearish_alignment and below_12h_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: bearish alignment forms or price breaks below 12h EMA
            if bearish_alignment or below_12h_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: bullish alignment forms or price breaks above 12h EMA
            if bullish_alignment or above_12h_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_WilliamsAlligator_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0