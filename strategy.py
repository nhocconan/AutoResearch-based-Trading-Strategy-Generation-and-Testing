#!/usr/bin/env python3
"""
1h_Donchian_20_4hTrend_1dVolume
Hypothesis: On 1h, enter long when price breaks above 20-period Donchian high with 4h uptrend (EMA50) and 1d volume spike (>2x average). Enter short when price breaks below 20-period Donchian low with 4h downtrend and 1d volume spike. Uses 4h for trend direction and 1d for volume confirmation, reducing false breakouts. Designed for 15-30 trades/year to avoid fee drag while capturing strong directional moves in both bull and bear markets.
"""
name = "1h_Donchian_20_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 4h uptrend + 1d volume spike
            if close[i] > donchian_high[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (vol_avg_1d_aligned[i] * 2.0):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low + 4h downtrend + 1d volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (vol_avg_1d_aligned[i] * 2.0):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion)
            if position == 1:
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals