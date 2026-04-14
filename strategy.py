# 4h_1d_camarilla_breakout_vol_filter_v2
# Hypothesis: Use 1d Camarilla pivot levels on 4h chart with volume confirmation to capture breakouts
# Camarilla levels (H4/L4) act as strong support/resistance - breakouts with volume indicate genuine moves
# Works in bull/bear markets: buy breakouts above H4, sell breakdowns below L4
# Volume filter ensures only significant breakouts trigger entries, reducing false signals
# Target: 20-50 trades/year on 4h timeframe to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    # Formula: Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    # Using (high + low + close) / 3 as pivot point approximation
    # Actual Camarilla uses previous day's close as base
    # Levels: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3/H2/H1 and L3/L2/L1 also calculated but we focus on H4/L4 for breakouts
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above H4 level with volume confirmation
            if price > h4_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below L4 level with volume confirmation
            elif price < l4_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below H4 level (failed breakout) or reverses below L4
            if price < h4_aligned[i] or price < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns above L4 level (failed breakdown) or reverses above H4
            if price > l4_aligned[i] or price > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_camarilla_breakout_vol_filter_v2"
timeframe = "4h"
leverage = 1.0