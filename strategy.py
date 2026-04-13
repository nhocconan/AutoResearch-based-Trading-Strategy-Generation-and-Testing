#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - Camarilla pivot breakout + volume confirmation
    # Works in bull/bear by trading strong intraday moves with volume validation
    # Target: 50-150 trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla levels (based on previous day)
    camarilla_h4 = (high_1d + low_1d + 2 * close_1d) / 3 + (high_1d - low_1d) * 1.1 / 6
    camarilla_l4 = (high_1d + low_1d + 2 * close_1d) / 3 - (high_1d - low_1d) * 1.1 / 6
    camarilla_h3 = (high_1d + low_1d + 2 * close_1d) / 3 + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = (high_1d + low_1d + 2 * close_1d) / 3 - (high_1d - low_1d) * 1.1 / 4
    camarilla_h2 = (high_1d + low_1d + 2 * close_1d) / 3 + (high_1d - low_1d) * 1.1 / 2
    camarilla_l2 = (high_1d + low_1d + 2 * close_1d) / 3 - (high_1d - low_1d) * 1.1 / 2
    camarilla_h1 = (high_1d + low_1d + 2 * close_1d) / 3 + (high_1d - low_1d) * 1.1 * 5 / 12
    camarilla_l1 = (high_1d + low_1d + 2 * close_1d) / 3 - (high_1d - low_1d) * 1.1 * 5 / 12
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h2_aligned[i]) or
            np.isnan(camarilla_l2_aligned[i]) or
            np.isnan(camarilla_h1_aligned[i]) or
            np.isnan(camarilla_l1_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-day average
        # Need to map 12h index to 1d index for volume
        vol_12h_idx = i // 2  # 2x 12h bars per 1d
        if vol_12h_idx >= len(volume):
            vol_12h_idx = len(volume) - 1
        vol_12h = volume[vol_12h_idx]
        vol_confirmed = vol_12h > 1.5 * vol_avg_20_aligned[i]
        
        # Breakout conditions using 12h close
        close_12h = close[i]
        
        # Long breakout: above H3 with volume
        enter_long = close_12h > camarilla_h3_aligned[i] and vol_confirmed
        # Short breakdown: below L3 with volume
        enter_short = close_12h < camarilla_l3_aligned[i] and vol_confirmed
        
        # Exit conditions: return to H4/L4 levels
        exit_long = position == 1 and close_12h <= camarilla_h4_aligned[i]
        exit_short = position == -1 and close_12h >= camarilla_l4_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
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