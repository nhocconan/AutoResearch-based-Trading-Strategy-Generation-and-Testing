#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and session filter (08-20 UTC).
# Donchian channels provide clear breakout levels. Volume > 1.5x 20-period average confirms breakout strength.
# Session filter reduces noise trades during low-volume hours. Designed for low trade frequency (~20-30/year)
# to minimize fee decay. Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Uses 4h for signal direction, 1h only for entry timing precision.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian calculation (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period Donchian channels
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe (waits for 4h bar to close)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_ma = vol_ma_aligned[i]
        vol = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation
            if price > upper_val and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian with volume confirmation
            elif price < lower_val and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower Donchian (breakdown)
                if price < lower_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper Donchian (breakout)
                if price > upper_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_Volume_Session"
timeframe = "1h"
leverage = 1.0