#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day trend filter and volume confirmation
# Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trend presence and direction
# In trending markets (JAW > TEETH > LIPS or JAW < TEETH < LIPS), trade Alligator direction
# In ranging markets (JAW, TEETH, LIPS intertwined), stay flat
# 1-day EMA50 confirms higher-timeframe trend bias
# Volume spike (>1.5x 20-period average) filters low-conviction moves
# Target: 20-40 trades/year to minimize fee decay while capturing strong trends

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-day EMA50 for trend bias ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Williams Alligator on 6h timeframe ===
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # === Volume Spike on 6h (vs 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator sleeping (intertwined) = ranging market
        jaw_teeth_close = abs(jaw_val - teeth_val) < (jaw_val * 0.001)
        teeth_lips_close = abs(teeth_val - lips_val) < (teeth_val * 0.001)
        jaws_lips_close = abs(jaw_val - lips_val) < (jaw_val * 0.001)
        is_ranging = jaw_teeth_close and teeth_lips_close and jaws_lips_close
        
        # Alligator awake - trending market
        is_bullish = jaw_val > teeth_val > lips_val
        is_bearish = jaw_val < teeth_val < lips_val
        
        # Volume spike filter
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        # 1-day EMA50 trend bias
        price_vs_ema50 = close[i] > ema50_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions: bullish Alligator + price above 1d EMA50 + volume spike
            if is_bullish and price_vs_ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short conditions: bearish Alligator + price below 1d EMA50 + volume spike
            elif is_bearish and not price_vs_ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when Alligator turns bearish or loses bullish alignment
            if not is_bullish:  # Either bearish or ranging
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Alligator turns bullish or loses bearish alignment
            if not is_bearish:  # Either bullish or ranging
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0