#!/usr/bin/env python3
"""
1h 4h Donchian breakout with 1d trend filter and volume confirmation
Long: price breaks above 4h Donchian high + price > 1d EMA200 + volume > 1.5x avg
Short: price breaks below 4h Donchian low + price < 1d EMA200 + volume > 1.5x avg
Exit: opposite breakout or volume drops below average
Designed for trend continuation with volume confirmation in all market regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_donchian_1d_trend_volume_v1"
timeframe = "1h"
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
    
    # === 4h Donchian Channels (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1d EMA200 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === Volume Filter (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or \
           np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h Donchian low OR volume drops
            if close[i] < donch_low_aligned[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian high OR volume drops
            if close[i] > donch_high_aligned[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: break above Donchian high + above 1d EMA200 + volume
            if close[i] > donch_high_aligned[i] and close[i] > ema_200_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.20
            # Short: break below Donchian low + below 1d EMA200 + volume
            elif close[i] < donch_low_aligned[i] and close[i] < ema_200_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.20
    
    return signals