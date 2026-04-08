#!/usr/bin/env python3
# 6h_1w_ema_donchian_volume_v1
# Hypothesis: Trade weekly EMA trend with Donchian channel breakouts and volume confirmation on 6h.
# In weekly uptrend (price > weekly EMA50): go long on breakout above 20-period Donchian high with volume surge.
# In weekly downtrend (price < weekly EMA50): go short on breakdown below 20-period Donchian low with volume surge.
# Exit when price returns to weekly EMA50 or opposite Donchian band is touched.
# Uses volume filter to avoid false breakouts. Target: 15-35 trades/year (60-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_ema_donchian_volume_v1"
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track holding period
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: price < weekly EMA50 or touches lower Donchian band
            # Minimum holding period: 3 bars (18 hours)
            if bars_since_entry >= 3 and (close[i] < ema50_1w_aligned[i] or close[i] <= donchian_low[i]):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: price > weekly EMA50 or touches upper Donchian band
            # Minimum holding period: 3 bars (18 hours)
            if bars_since_entry >= 3 and (close[i] > ema50_1w_aligned[i] or close[i] >= donchian_high[i]):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            bars_since_entry = 0
            # Long entry: weekly uptrend + breakout above Donchian high + volume surge
            if (close[i] > ema50_1w_aligned[i] and 
                close[i] > donchian_high[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly downtrend + breakdown below Donchian low + volume surge
            elif (close[i] < ema50_1w_aligned[i] and 
                  close[i] < donchian_low[i] and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals