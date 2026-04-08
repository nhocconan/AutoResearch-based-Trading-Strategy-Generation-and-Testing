#!/usr/bin/env python3
# 6h_1d_1w_price_channel_breakout_volume
# Hypothesis: Combine daily price channel (Donchian 20) with weekly trend filter and volume confirmation on 6h.
# In bull markets: buy breakouts above daily Donchian high when weekly trend is up.
# In bear markets: sell breakdowns below daily Donchian low when weekly trend is down.
# Weekly trend defined by price above/below weekly EMA50. Volume confirmation reduces false breakouts.
# Target: 15-30 trades/year on 6h timeframe with strict breakout conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_price_channel_breakout_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling max/min for Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 6h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below daily Donchian low OR weekly trend turns down
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above daily Donchian high OR weekly trend turns up
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above daily Donchian high with volume surge and weekly uptrend
            if close[i] > donchian_high_aligned[i] and vol_surge and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily Donchian low with volume surge and weekly downtrend
            elif close[i] < donchian_low_aligned[i] and vol_surge and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals