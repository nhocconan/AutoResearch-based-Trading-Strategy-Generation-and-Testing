#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume regime filter
# - Long when price breaks above H3 level AND 1d volume > 1.2x 20-period average
# - Short when price breaks below L3 level AND 1d volume > 1.2x 20-period average
# - Exit when price crosses Pivot Point (PP) or opposite pivot break occurs
# - Camarilla levels provide statistically significant intraday support/resistance
# - 1d volume regime ensures we trade only during institutional participation
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years) to avoid fee drag

name = "4h_1d_camarilla_pivot_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    h3 = pp + (range_hl * 1.1 / 4)
    l3 = pp - (range_hl * 1.1 / 4)
    h4 = pp + (range_hl * 1.1 / 2)
    l4 = pp - (range_hl * 1.1 / 2)
    
    # Align HTF levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Pre-compute 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=float)
    for i in range(19, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    vol_4h = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume regime filter: 1d volume > 1.2x 20-period average
        vol_regime = vol_4h[i] > 1.2 * vol_ma_20_aligned[i]
        
        close_now = close_4h[i]
        h3_now = h3_aligned[i]
        l3_now = l3_aligned[i]
        pp_now = pp_aligned[i]
        
        # Camarilla breakout signals
        break_h3 = close_now > h3_now  # price breaks above H3
        break_l3 = close_now < l3_now  # price breaks below L3
        cross_pp_up = (close_4h[i-1] <= pp_now and close_now > pp_now)  # crosses above PP
        cross_pp_down = (close_4h[i-1] >= pp_now and close_now < pp_now)  # crosses below PP
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume regime
            if (break_h3 and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume regime
            elif (break_l3 and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses PP or opposite pivot break
            exit_long = (position == 1 and 
                        (cross_pp_down or break_l3))
            exit_short = (position == -1 and 
                         (cross_pp_up or break_h3))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals