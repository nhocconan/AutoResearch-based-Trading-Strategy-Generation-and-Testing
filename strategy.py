#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot point reversal strategy.
# Uses weekly pivot points (calculated from prior week's OHLC) for support/resistance levels.
# Enters long when price bounces above weekly pivot support with volume confirmation,
# enters short when price rejects below weekly pivot resistance with volume confirmation.
# Exits when price reaches opposite pivot level or shows reversal signs.
# Designed to work in both bull and bear markets by fading extremes at key weekly levels.
# Targets 12-37 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot point calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    s1 = 2 * pivot - high_1w  # Support 1
    r1 = 2 * pivot - low_1w   # Resistance 1
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Load daily data for additional context and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        pivot_val = pivot_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        
        if position == 0:
            # Long entry: price bounces above S1 support with volume confirmation
            if price > s1_val and price <= pivot_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price rejects below R1 resistance with volume confirmation
            elif price < r1_val and price >= pivot_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches pivot level or shows rejection
                if price >= pivot_val:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit when price reaches pivot level or shows bounce
                if price <= pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_S1R1_Bounce"
timeframe = "6h"
leverage = 1.0