#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h trend continuation at 1d Camarilla H3/L3 breakout with volume confirmation
    # Uses 1d Camarilla levels for structure, 12h for entry timing precision
    # Works in bull/bear: breakouts capture momentum in trending markets, volume filters false signals
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels (H3/L3 are breakout levels, H4/L4 are stronger continuation levels)
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
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
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        idx_1d = i // 2
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions at Camarilla H3/L3 levels
        breakout_long = close[i] > camarilla_h3_aligned[i]  # Price above H3 -> long breakout
        breakout_short = close[i] < camarilla_l3_aligned[i]  # Price below L3 -> short breakout
        
        # Exit conditions: price returns to opposite H4/L4 levels (mean reversion to stronger levels)
        exit_long = position == 1 and close[i] <= camarilla_l4_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_h4_aligned[i]
        
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

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0