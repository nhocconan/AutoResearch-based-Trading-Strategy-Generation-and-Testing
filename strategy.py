#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with Williams %R extremes on 1d combined with volume spike and 12h EMA trend filter.
Long when Williams %R < -80 (oversold) + volume > 2x 24-period average + 12h EMA34 > EMA89.
Short when Williams %R > -20 (overbought) + volume > 2x 24-period average + 12h EMA34 < EMA89.
Williams %R identifies exhaustion points; volume spike confirms participation; 12h EMA filter ensures alignment with medium-term trend.
Targets mean reversion in both bull and bear markets - oversold bounces in bear, overbought pullbacks in bull.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL)
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d volume 24-period average
    vol_ma_24_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA89
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align all to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_24_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema89_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 24-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_24_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + volume spike + bullish 12h trend (EMA34 > EMA89)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                ema34_12h_aligned[i] > ema89_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + volume spike + bearish 12h trend (EMA34 < EMA89)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  ema34_12h_aligned[i] < ema89_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend turns bearish
            if (williams_r_aligned[i] > -50 or 
                ema34_12h_aligned[i] < ema89_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend turns bullish
            if (williams_r_aligned[i] < -50 or 
                ema34_12h_aligned[i] > ema89_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_VolumeSpike_12hEMA"
timeframe = "6h"
leverage = 1.0