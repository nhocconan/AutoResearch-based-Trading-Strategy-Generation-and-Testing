#!/usr/bin/env python3
# 12h_1w_1d_momentum_v1
# Hypothesis: Trade momentum breakouts on 12h timeframe with weekly trend filter (1w EMA50) and daily volume confirmation.
# In weekly uptrend: go long on breakout above 12h Donchian high (20) with volume surge.
# In weekly downtrend: go short on breakdown below 12h Donchian low (20) with volume surge.
# Uses weekly EMA50 for trend, 12h Donchian channels for breakout signals, and volume > 2x 20-period average for confirmation.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 12h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Daily volume confirmation: volume > 2x 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or weekly trend breaks
            if close[i] < donchian_low[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or weekly trend breaks
            if close[i] > donchian_high[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume surge and weekly uptrend
            if (close[i] > donchian_high[i] and vol_surge and 
                close[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume surge and weekly downtrend
            elif (close[i] < donchian_low[i] and vol_surge and 
                  close[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals