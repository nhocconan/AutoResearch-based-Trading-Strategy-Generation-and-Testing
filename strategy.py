#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using weekly Camarilla pivot breakouts with 1d volume confirmation
    # Long when price breaks above weekly R4 with 1d volume spike
    # Short when price breaks below weekly S4 with 1d volume spike
    # Exit when price returns to weekly H3/L3 levels
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
    # Works in both bull and bear: breakouts capture strong moves, volume confirmation filters false signals
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for HTF Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (based on previous 1w bar)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Calculate pivot point (PP)
    pp = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels
    camarilla_h3 = pp + 1.125 * range_1w
    camarilla_l3 = pp - 1.125 * range_1w
    camarilla_h4 = pp + 1.5 * range_1w
    camarilla_l4 = pp - 1.5 * range_1w
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        # Get the 1d bar index for current 6h bar (each 1d bar = 4 6h bars)
        idx_1d = i // 4
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions at Camarilla H4/L4 levels
        breakout_long = close[i] > camarilla_h4_aligned[i]  # Price above H4 -> long breakout
        breakout_short = close[i] < camarilla_l4_aligned[i]  # Price below L4 -> short breakout
        
        # Exit conditions: price returns to Camarilla H3/L3 levels
        exit_long = position == 1 and close[i] <= camarilla_h3_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_l3_aligned[i]
        
        # Execute signals
        if breakout_long and volume_confirmed and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_confirmed and position != -1:
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

name = "6h_1w_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0