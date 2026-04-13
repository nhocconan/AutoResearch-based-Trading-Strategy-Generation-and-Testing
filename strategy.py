#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Long: price breaks above R4 + 1d volume > 1.5x 20-period average
    # Short: price breaks below S4 + 1d volume > 1.5x 20-period average
    # Exit: price returns to daily pivot point (PP)
    # Uses Camarilla levels from prior 1d for structure, volume for confirmation
    # Breakouts with volume are reliable in both bull and bear markets
    # Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for Camarilla pivots and volume (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate prior 1d Camarilla levels
    # Based on prior day's range: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # PP = (H+L+C)/3
    prior_high_1d = np.concatenate([[np.nan], high_1d[:-1]])  # shift by 1 bar
    prior_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prior_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    cam_pp = (prior_high_1d + prior_low_1d + prior_close_1d) / 3
    cam_range = prior_high_1d - prior_low_1d
    cam_r4 = cam_pp + cam_range * 1.1 / 2
    cam_s4 = cam_pp - cam_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    cam_pp_aligned = align_htf_to_ltf(prices, df_1d, cam_pp)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for volume average
        # Skip if data not ready
        if (np.isnan(cam_r4_aligned[i]) or np.isnan(cam_s4_aligned[i]) or 
            np.isnan(cam_pp_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Get current 1d volume (aligned)
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 1.5 * vol_avg_20_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > cam_r4_aligned[i] and volume_confirmed
        breakout_short = close[i] < cam_s4_aligned[i] and volume_confirmed
        
        # Exit conditions: return to daily pivot point
        exit_long = position == 1 and close[i] <= cam_pp_aligned[i]
        exit_short = position == -1 and close[i] >= cam_pp_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0