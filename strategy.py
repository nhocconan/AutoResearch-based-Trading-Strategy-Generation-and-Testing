#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1w HTF - Weekly Camarilla pivot breakout with volume confirmation
    # Weekly Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) provide institutional reference points
    # Volume confirmation filters false breakouts. Works in both bull (breakout continuation) and bear (mean reversion at extremes)
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get weekly data for HTF Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for volume context (more stable than 6h volume alone)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate weekly Camarilla pivot levels (based on prior week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    typical_price = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    camarilla_r4 = close_1w + range_1w * 1.1 / 2
    camarilla_r3 = close_1w + range_1w * 1.1 / 4
    camarilla_s3 = close_1w - range_1w * 1.1 / 4
    camarilla_s4 = close_1w - range_1w * 1.1 / 2
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x 20-period 1d average
        # Using 1d volume average for stability vs noisy 6h volume
        volume_confirmed = volume[i] > 1.2 * vol_avg_20_aligned[i]
        
        # Mean reversion at extreme levels (R3/S3) - fade the extreme
        mean_revert_long = close[i] <= camarilla_s3_aligned[i] and volume_confirmed
        mean_revert_short = close[i] >= camarilla_r3_aligned[i] and volume_confirmed
        
        # Breakout continuation at extreme levels (R4/S4) - break the extreme
        breakout_long = close[i] >= camarilla_r4_aligned[i] and volume_confirmed
        breakout_short = close[i] <= camarilla_s4_aligned[i] and volume_confirmed
        
        # Exit conditions: return to pivot area (mean reversion) or opposite extreme (breakout)
        exit_long = position == 1 and (close[i] <= camarilla_r3_aligned[i] or close[i] >= camarilla_s4_aligned[i])
        exit_short = position == -1 and (close[i] >= camarilla_s3_aligned[i] or close[i] <= camarilla_r4_aligned[i])
        
        # Execute signals - prioritize breakout over mean reversion when both triggered
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif mean_revert_long and position != 1 and position != -1:  # only if flat
            position = 1
            signals[i] = position_size
        elif mean_revert_short and position != 1 and position != -1:  # only if flat
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

name = "6h_1w_camarilla_breakout_meanrev_v1"
timeframe = "6h"
leverage = 1.0