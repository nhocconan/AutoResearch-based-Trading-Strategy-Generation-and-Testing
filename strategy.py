#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h primary with 4h/1d HTF - Camarilla pivot breakout + volume spike + session filter
    # Uses Camarilla levels (H3/L3) from 4h for structure, 1d volume spike for conviction, 
    # and 08-20 UTC session to avoid low-liquidity hours. Works in bull/bear by trading
    # institutional breakouts during active hours. Target: 60-150 trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Get 4h data for Camarilla pivot calculation (structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation (conviction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 4h bar (H3, L3, H4, L4)
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    camarilla_high = np.maximum(high_4h, np.roll(high_4h, 1))
    camarilla_low = np.minimum(low_4h, np.roll(low_4h, 1))
    camarilla_close = np.roll(close_4h, 1)
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_h3 = camarilla_close + camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_close - camarilla_range * 1.1 / 4
    camarilla_h4 = camarilla_close + camarilla_range * 1.1 / 2
    camarilla_l4 = camarilla_close - camarilla_range * 1.1 / 2
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size (discrete level)
    
    for i in range(50, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 1h volume > 2.0x 20-day average
        volume_spike = volume[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Breakout conditions using Camarilla H3/L3
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_spike
        enter_short = breakout_down and volume_spike
        
        # Exit conditions: reverse at H4/L4 levels or opposite H3/L3 break
        exit_long = (position == 1 and 
                    (close[i] >= camarilla_h4_aligned[i] or 
                     close[i] < camarilla_l3_aligned[i]))
        exit_short = (position == -1 and 
                     (close[i] <= camarilla_l4_aligned[i] or 
                      close[i] > camarilla_h3_aligned[i]))
        
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

name = "1h_4h_1d_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0