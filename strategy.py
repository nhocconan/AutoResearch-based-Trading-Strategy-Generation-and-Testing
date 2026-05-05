#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# Long when: Price breaks above Donchian upper channel (20) AND 1d volume > 1.5 * 20-period average AND price > 1w EMA50
# Short when: Price breaks below Donchian lower channel (20) AND 1d volume > 1.5 * 20-period average AND price < 1w EMA50
# Exit when price returns to Donchian middle (mean of upper/lower)
# Donchian breakout captures strong directional moves after consolidation
# Volume spike confirms institutional participation
# 1w EMA50 filter ensures we only trade in direction of higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_Donchian20_1dVolumeSpike_1wEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 1.5 * 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian Channels (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donchian = highest_20
    lower_donchian = lowest_20
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or np.isnan(middle_donchian[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike and trend filters
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Boolean as float
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike and price above 1w EMA50
            if close[i] > upper_donchian[i] and vol_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume spike and price below 1w EMA50
            elif close[i] < lower_donchian[i] and vol_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian (mean reversion)
            if close[i] < middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian (mean reversion)
            if close[i] > middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals