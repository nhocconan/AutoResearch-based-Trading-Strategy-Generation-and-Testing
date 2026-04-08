#!/usr/bin/env python3
# 1d_1w_price_channel_breakout_volume_v1
# Hypothesis: Use 1w Donchian channel breakouts with 1d volume confirmation for trend following.
# Works in bull markets (breakouts above upper channel) and bear markets (breakouts below lower channel).
# Targets 15-25 trades/year by requiring weekly channel breaks + volume surge to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_price_channel_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper channel: highest high of last 20 weeks
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 weeks
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume confirmation: volume > 2.0x average of last 20 days (approx 1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly lower channel
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly upper channel
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly upper channel with volume confirmation
            if close[i] > donchian_upper_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly lower channel with volume confirmation
            elif close[i] < donchian_lower_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals