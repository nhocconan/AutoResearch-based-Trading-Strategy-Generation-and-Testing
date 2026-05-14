#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h pivot-based breakout and volume confirmation.
# Uses 12h Camarilla pivot levels (R4/S4 for breakout, R3/S3 for fade) with volume spike.
# Designed to work in both bull and bear markets by capturing institutional breakouts
# while avoiding false moves in low-volume conditions. Targets 12-37 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Camarilla pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels for 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r4_12h = pp_12h + (range_12h * 1.1 / 2)
    r3_12h = pp_12h + (range_12h * 1.1 / 4)
    s3_12h = pp_12h - (range_12h * 1.1 / 4)
    s4_12h = pp_12h - (range_12h * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe (wait for 12h bar close)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter)
        vol_spike = vol > 2.0 * vol_ma
        
        r4 = r4_12h_aligned[i]
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume spike
            if price > r4 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume spike
            elif price < s4 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to R3 (take profit) or breakdown below S4 (stop)
                if price < r3 or price < s4:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to S3 (take profit) or breakout above R4 (stop)
                if price > s3 or price > r4:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0