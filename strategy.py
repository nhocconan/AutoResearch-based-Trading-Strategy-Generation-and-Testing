#!/usr/bin/env python3
"""
4H_DONCHIAN_BREAKOUT_1DVOLUME_CONFIRMATION_1W_VOLUME_REGIME
Hypothesis: Donchian(20) breakout on 4h with 1d volume confirmation and 1w volume regime filter. 
Breakouts only when volume > 1.5x 20-day average AND weekly volume > 50th percentile. 
This filters out low-conviction breakouts and focuses on institutional participation. 
Exit when price crosses 20-period EMA on 4h. Designed for fewer, higher-quality trades.
"""
name = "4H_DONCHIAN_BREAKOUT_1DVOLUME_CONFIRMATION_1W_VOLUME_REGIME"
timeframe = "4h"
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
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_20d_ratio = vol_1d / vol_ma_20d  # Current day volume / 20-day average
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_20d_ratio)
    
    # 1w data for volume regime (filter for high-volume weeks)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    vol_1w = df_1w['volume'].values
    vol_percentile_50w = pd.Series(vol_1w).rolling(window=50, min_periods=30).quantile(0.5).values
    vol_1w_above_median = vol_1w > vol_percentile_50w
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_above_median, additional_delay_bars=1)
    
    # Exit condition: 20-period EMA on 4h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(vol_regime_aligned[i]) or
            np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above Donchian high with volume confirmation and regime filter
            if (high[i] > high_max[i] and 
                vol_ratio_aligned[i] > 1.5 and 
                vol_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low with volume confirmation and regime filter
            elif (low[i] < low_min[i] and 
                  vol_ratio_aligned[i] > 1.5 and 
                  vol_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 20 EMA
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 20 EMA
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals