#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - Camarilla H3/L3 mean reversion with volume spike
    # In ranging markets (common in 2025 BTC/ETH), price tends to revert from extreme Camarilla levels
    # Uses volume spike to confirm institutional interest at these levels
    # Target: 50-100 trades over 4 years (12-25/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation (use same timeframe as primary)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate previous day's Camarilla levels (H3/L3 for mean reversion)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Calculate 4h volume average (20-period) for spike detection
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
        volume_confirmed = volume_4h[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Mean reversion conditions at Camarilla H3/L3 levels
        touch_h3 = close[i] >= camarilla_h3_aligned[i]  # Price touches or exceeds upper level
        touch_l3 = close[i] <= camarilla_l3_aligned[i]  # Price touches or exceeds lower level
        
        # Entry conditions: fade the extreme level with volume confirmation
        enter_long = touch_l3 and volume_confirmed  # Long when price touches L3
        enter_short = touch_h3 and volume_confirmed  # Short when price touches H3
        
        # Exit conditions: return to previous day's close (mean reversion target)
        exit_long = position == 1 and close[i] >= prev_close_1d_aligned if 'prev_close_1d_aligned' in locals() else close[i] >= np.roll(close_1d, 1)[i] if i < len(np.roll(close_1d, 1)) else False
        exit_short = position == -1 and close[i] <= prev_close_1d_aligned if 'prev_close_1d_aligned' in locals() else close[i] <= np.roll(close_1d, 1)[i] if i < len(np.roll(close_1d, 1)) else False
        
        # Simplify exit: exit when price reaches midpoint between H3/L3 (more realistic)
        camarilla_mid = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
        exit_long = position == 1 and close[i] >= camarilla_mid
        exit_short = position == -1 and close[i] <= camarilla_mid
        
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

name = "4h_1d_camarilla_h3l3_meanrev_volume_v1"
timeframe = "4h"
leverage = 1.0