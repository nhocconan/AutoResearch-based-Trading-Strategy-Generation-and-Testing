#!/usr/bin/env python3
"""
6h Williams %R Mean Reversion with 1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025-2026 bear/range conditions), Williams %R
identifies overbought/oversold conditions. Combined with 1d EMA trend filter to avoid
counter-trend trades and volume spike for confirmation, this captures mean reversion
bounces with controlled risk. Works in both bull (pullbacks in uptrend) and bear
(bounces in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_r_mean_reversion_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter (smooth trend)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, oversold < -80, overbought > -20
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume spike: current volume > 2.0 x 20-period average (strong participation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral (> -50) OR trend turns bearish
            if williams_r[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (< -50) OR trend turns bullish
            if williams_r[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for mean reversion entry
            # Long: Williams %R oversold (< -80) + volume spike + uptrend bias
            if (williams_r[i] < -80 and
                vol_spike[i] and
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought (> -20) + volume spike + downtrend bias
            elif (williams_r[i] > -20 and
                  vol_spike[i] and
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals