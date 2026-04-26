#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla pivot R1/S1 breakout on 1h with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period MA). 
Long when price breaks above R1 in 4h uptrend with volume spike. 
Short when price breaks below S1 in 4h downtrend with volume spike. 
Uses discrete position sizing (0.20) to minimize fee churn. 
Designed to work in both bull and bear markets by following the 4h trend. 
Target: 15-37 trades/year (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar's OHLC)
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    typical_price_4h_vals = typical_price_4h.values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    camarilla_range = (high_4h - low_4h) * 1.1 / 12
    r1_4h = typical_price_4h_vals + camarilla_range
    s1_4h = typical_price_4h_vals - camarilla_range
    
    # Align Camarilla levels to 1h timeframe (previous completed 4h bar)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: volume > 1.5x 20-period MA on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA + 20 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 in 4h uptrend with volume spike
            if (close[i] > r1_4h_aligned[i] and uptrend_4h[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in 4h downtrend with volume spike
            elif (close[i] < s1_4h_aligned[i] and downtrend_4h[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: 4h trend changes to downtrend OR price breaks below S1 (failed breakout)
            if (not uptrend_4h[i] or close[i] < s1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: 4h trend changes to uptrend OR price breaks above R1 (failed breakout)
            if (not downtrend_4h[i] or close[i] > r1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0