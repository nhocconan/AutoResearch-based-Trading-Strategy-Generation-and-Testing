#!/usr/bin/env python3
# 6h_1w_1d_price_channel_breakout_volume_v1
# Hypothesis: 6-hour price channel breakouts (Donchian 20) filtered by weekly EMA20 trend and daily volume confirmation.
# The 6-hour timeframe captures intermediate trends while avoiding excessive noise. Weekly EMA20 provides strong trend filter
# to avoid countertrend trades, and daily volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by only trading in direction of weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_price_channel_breakout_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 6h Donchian high, above weekly EMA20, with volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema20_1w_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 6h Donchian low, below weekly EMA20, with volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema20_1w_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals