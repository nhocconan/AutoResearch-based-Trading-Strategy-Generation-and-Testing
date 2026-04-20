#!/usr/bin/env python3
# 12h_FundingRate_MeanReversion_with_VolumeFilter
# Hypothesis: Funding rate mean reversion on 12h timeframe with weekly and daily trend filters.
# Extreme funding rates (positive for shorts, negative for longs) indicate overextended sentiment.
# In bull markets (price > weekly EMA50): short when funding > 0.03%, long when funding < -0.03%.
# In bear markets (price < weekly EMA50): reverse logic for counter-trend mean reversion.
# Volume confirmation ensures institutional participation. Max 0.30 position size.
# Target: 15-35 trades/year to minimize fee drag in ranging 2025+ markets.

name = "12h_FundingRate_MeanReversion_with_VolumeFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter and funding rate data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily average volume for volume filter (20-period)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Funding rate data (using open price as proxy - actual funding would come from separate data)
    # For this implementation, we'll use price deviation from weekly VWAP as funding proxy
    typical_price = (high + low + close) / 3
    vwap_1w = (typical_price * volume).cumsum() / volume.cumsum()
    # Resample to weekly and align (simplified funding proxy)
    funding_proxy = (close - vwap_1w) / vwap_1w  # Normalized deviation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_20_aligned[i]) or 
            np.isnan(funding_proxy[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume must be above average
        volume_ok = volume[i] > avg_volume_20_aligned[i]
        
        if position == 0:
            # Determine trend from weekly EMA50
            uptrend = close[i] > ema50_1w_aligned[i]
            downtrend = close[i] < ema50_1w_aligned[i]
            
            # Extreme funding signals for mean reversion
            funding_extreme_long = funding_proxy[i] < -0.0003  # -0.03% threshold
            funding_extreme_short = funding_proxy[i] > 0.0003   # +0.03% threshold
            
            # In uptrend: look for extreme negative funding (oversold) to go long
            # In downtrend: look for extreme positive funding (overbought) to go short
            if uptrend and funding_extreme_long and volume_ok:
                signals[i] = 0.30
                position = 1
            elif downtrend and funding_extreme_short and volume_ok:
                signals[i] = -0.30
                position = -1
                
        elif position == 1:
            # Long: exit when funding normalizes or trend breaks
            if (funding_proxy[i] > -0.0001 or  # Funding back to neutral
                close[i] < ema50_1w_aligned[i] or  # Trend break
                volume[i] < avg_volume_20_aligned[i] * 0.5):  # Volume collapse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:
            # Short: exit when funding normalizes or trend breaks
            if (funding_proxy[i] < 0.0001 or  # Funding back to neutral
                close[i] > ema50_1w_aligned[i] or  # Trend break
                volume[i] < avg_volume_20_aligned[i] * 0.5):  # Volume collapse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals