# 6h Weekly Pivot Breakout with Daily Volume Confirmation
# Hypothesis: Weekly pivot points act as key institutional support/resistance.
# Breakouts above R2 or below S2 with strong daily volume (>1.5x 20-period average)
# indicate institutional participation and continuation. Works in bull/bear markets
# by capturing breakouts from key levels. Target: 15-25 trades/year per symbol.

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points - call ONCE before loop
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Get daily data for volume filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    volume_d = df_d['volume'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pp_w = (high_w + low_w + close_w) / 3
    r2_w = pp_w + (high_w - low_w)  # R2 = P + (H - L)
    s2_w = pp_w - (high_w - low_w)  # S2 = P - (H - L)
    
    # Calculate 20-period average volume for daily timeframe
    vol_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Get aligned weekly pivot values for current 6h bar
        r2 = align_htf_to_ltf(prices, df_w, r2_w)[i]
        s2 = align_htf_to_ltf(prices, df_w, s2_w)[i]
        
        # Get aligned daily volume average for current 6h bar
        vol_ma = align_htf_to_ltf(prices, df_d, vol_ma_d)[i]
        
        # Skip if any required data is NaN
        if np.isnan(r2) or np.isnan(s2) or np.isnan(vol_ma) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma
        
        if position == 1:  # Long position
            # Exit if price breaks below S2 (failed breakout)
            if close[i] < s2:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R2 (failed breakout)
            if close[i] > r2:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above R2 with volume confirmation
            if close[i] > r2 and vol_breakout:
                position = 1
                signals[i] = 0.25
            # Breakout short below S2 with volume confirmation
            elif close[i] < s2 and vol_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals