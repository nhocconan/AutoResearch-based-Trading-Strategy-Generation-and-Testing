# 1d_1w_donchian_breakout_volume_v1
# Hypothesis: Weekly Donchian breakout with daily volume confirmation targets strong trends while avoiding false breakouts in choppy markets. Weekly timeframe reduces trade frequency; volume filter ensures momentum validity.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals