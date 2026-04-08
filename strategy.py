#!/usr/bin/env python3
# 12h_camarilla_pivot_daily_trend_volume_v1
# Hypothesis: Uses Camarilla pivot levels from 1-day timeframe with volume confirmation and trend filter. 
# Long when price breaks above resistance level (H4) with volume, short when breaks below support (L4).
# Includes trend filter (price vs daily EMA200) to avoid counter-trend trades. Designed for low frequency (12-37 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    H4 = typical_price + 1.1 * (high_1d - low_1d) / 2
    L4 = typical_price - 1.1 * (high_1d - low_1d) / 2
    H3 = typical_price + 1.1 * (high_1d - low_1d) / 4
    L3 = typical_price - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Need EMA200 warmed up
    
    for i in range(start_idx, n):
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price below L3 or trend reversal
            if close[i] < L3_aligned[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price above H3 or trend reversal
            if close[i] > H3_aligned[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in direction of daily trend
            if close[i] > ema200_1d_aligned[i]:  # Uptrend
                # Long breakout above H4 with volume
                if close[i] > H4_aligned[i] and volume_ok:
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                # Short breakdown below L4 with volume
                if close[i] < L4_aligned[i] and volume_ok:
                    position = -1
                    signals[i] = -0.25
    
    return signals