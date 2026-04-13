#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation
    # Long when: price breaks above R4 AND 12h volume > 1.5x 20-period average
    # Short when: price breaks below S4 AND 12h volume > 1.5x 20-period average
    # Exit when: price returns to daily pivot point (PP)
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Camarilla levels from 1d provide institutional structure; volume confirms momentum.
    # Works in bull/bear via breakout logic that captures strong moves in either direction.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    PP = (high_1d + low_1d + close_1d) / 3
    S1 = PP - (high_1d - low_1d) * 1.1 / 12
    S2 = PP - (high_1d - low_1d) * 1.1 / 6
    S3 = PP - (high_1d - low_1d) * 1.1 / 4
    S4 = PP - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['taker_buy_volume'].values  # using taker_buy_volume as proxy for volume
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        # Get the raw 12h volume value aligned to current 6h bar
        vol_12h_idx = i // 2  # 2x 6h bars per 12h bar (approximate for alignment check)
        if vol_12h_idx < len(df_12h):
            vol_12h_current = vol_12h[vol_12h_idx]
        else:
            vol_12h_current = vol_12h[-1] if len(vol_12h) > 0 else 0
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirmed = vol_12h_current > (vol_ma_aligned[i] * 1.5)
        
        # Breakout conditions
        breakout_long = close[i] > R4_aligned[i] and volume_confirmed
        breakout_short = close[i] < S4_aligned[i] and volume_confirmed
        
        # Exit conditions: return to pivot point
        exit_long = position == 1 and close[i] <= PP_aligned[i]
        exit_short = position == -1 and close[i] >= PP_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_12h_volume_v1"
timeframe = "6h"
leverage = 1.0