#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - 1d Camarilla pivot breakout with 4h volume confirmation
    # Camarilla pivots provide institutional support/resistance levels that work in both bull and bear markets
    # Volume confirmation ensures breakouts have conviction, reducing false signals
    # Target: 75-200 trades over 4 years (19-50/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    #            H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    #            H2 = Close + 0.75*(High-Low), L2 = Close - 0.75*(High-Low)
    #            H1 = Close + 0.5*(High-Low), L1 = Close - 0.5*(High-Low)
    #            Pivot = (High + Low + Close)/3
    # We'll use H3/L3 and H4/L4 as breakout levels
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Calculate 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF/LTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume_4h[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Breakout conditions using Camarilla H3/L3 and H4/L4
        breakout_up = close_4h[i] > camarilla_h3_aligned[i]
        breakout_down = close_4h[i] < camarilla_l3_aligned[i]
        
        # Strong breakout conditions using H4/L4 (more significant levels)
        strong_breakout_up = close_4h[i] > camarilla_h4_aligned[i]
        strong_breakout_down = close_4h[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed
        enter_short = breakout_down and volume_confirmed
        
        # Exit conditions: price returns to opposite Camarilla level or strong reversal
        exit_long = position == 1 and (close_4h[i] <= camarilla_l3_aligned[i] or strong_breakout_down)
        exit_short = position == -1 and (close_4h[i] >= camarilla_h3_aligned[i] or strong_breakout_up)
        
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

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0