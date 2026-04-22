#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load daily data once for HL2 and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate HL2 (typical price) for Elder Ray
    hl2_1d = (high_1d + low_1d) / 2
    
    # Calculate EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    hl2_aligned = align_htf_to_ltf(prices, df_1d, hl2_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any data is not ready
        if (np.isnan(hl2_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        hl2 = hl2_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Calculate Elder Ray components
        bull_power = price - ema50
        bear_power = ema50 - price
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Bull Power > 0 + volume spike + price > HL2
            if bull_power > 0 and vol_spike and price > hl2:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 + volume spike + price < HL2
            elif bear_power > 0 and vol_spike and price < hl2:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Elder Ray reverses or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power <= 0 or volume dries up
                if bull_power <= 0 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power <= 0 or volume dries up
                if bear_power <= 0 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_EMA50_Volume"
timeframe = "6h"
leverage = 1.0