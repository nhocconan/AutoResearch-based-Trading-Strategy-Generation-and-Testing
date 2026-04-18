#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w trend filter.
# Long when price breaks above 4h Donchian upper channel + 1d volume > 2x 20-period average + price > 1w EMA(50).
# Short when price breaks below 4h Donchian lower channel + 1d volume > 2x 20-period average + price < 1w EMA(50).
# Exit when price returns to 4h Donchian middle (mean of upper and lower).
# Designed for ~20-40 trades/year per symbol.
name = "4h_Donchian20_1dVolumeSpike_1wEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_max_20
    donchian_lower = low_min_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        
        # Volume spike condition: current 1d volume > 2x 20-period average
        volume_spike = volume[i] > (2.0 * vol_ma_val) if not np.isnan(vol_ma_val) else False
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike and above 1w EMA
            if close_val > upper and volume_spike and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike and below 1w EMA
            elif close_val < lower and volume_spike and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle channel
            if close_val <= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle channel
            if close_val >= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals