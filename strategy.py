#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) representing market equilibrium
# Long: Lips > Teeth > Jaw (green alignment) AND price above 1d EMA50 with volume spike
# Short: Lips < Teeth < Jaw (red alignment) AND price below 1d EMA50 with volume spike
# Works in trending markets by identifying when trend is aligned and strong
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Williams Alligator: three SMAs
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = close_s.rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 30-period average (stricter for lower trade frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # Need enough data for Alligator and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Green alignment: Lips > Teeth > Jaw (bullish trend)
            green_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Red alignment: Lips < Teeth < Jaw (bearish trend)
            red_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Green alignment AND price above 1d EMA50 AND volume spike
            if green_alignment and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Red alignment AND price below 1d EMA50 AND volume spike
            elif red_alignment and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks OR price crosses below 1d EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR price crosses above 1d EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals