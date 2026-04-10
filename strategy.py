#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly pivot direction filter
# - Uses 1d Camarilla H3/L3 levels for breakout entries
# - Uses 1w Camarilla H3/L3 levels to determine higher timeframe bias
# - Only take longs when price > weekly H3 (bullish bias) and price breaks above 1d H3
# - Only take shorts when price < weekly L3 (bearish bias) and price breaks below 1d L3
# - Volume confirmation: current volume > 2.0x 20-period average to filter weak breakouts
# - Exit: opposite 1d Camarilla level touch (H3 for shorts, L3 for longs)
# - Position size: 0.25 (25% of capital)
# - Target: 15-30 trades/year on 6h (60-120 total over 4 years) to minimize fee drag
# - Works in both bull/bear: weekly pivot filter adapts to higher timeframe regime

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_hl_1d = high_1d - low_1d
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    h3_1d = close_1d + range_hl_1d * 1.1 / 4.0
    l3_1d = close_1d - range_hl_1d * 1.1 / 4.0
    
    # Pre-compute 1w Camarilla pivot levels (H3, L3) for bias filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and ranges for 1w
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_hl_1w = high_1w - low_1w
    
    # Camarilla levels for 1w
    h3_1w = close_1w + range_hl_1w * 1.1 / 4.0
    l3_1w = close_1w - range_hl_1w * 1.1 / 4.0
    
    # Align HTF levels to LTF
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Pre-compute 6h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > 1d H3 AND price > weekly H3 (bullish bias) AND volume confirmation
            if (prices['close'].iloc[i] > h3_1d_aligned[i] and 
                prices['close'].iloc[i] > h3_1w_aligned[i] and 
                volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < 1d L3 AND price < weekly L3 (bearish bias) AND volume confirmation
            elif (prices['close'].iloc[i] < l3_1d_aligned[i] and 
                  prices['close'].iloc[i] < l3_1w_aligned[i] and 
                  volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price touches opposite 1d Camarilla level
            exit_long = prices['close'].iloc[i] < l3_1d_aligned[i]   # Price breaks below 1d L3 (exit long)
            exit_short = prices['close'].iloc[i] > h3_1d_aligned[i]  # Price breaks above 1d H3 (exit short)
            
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals