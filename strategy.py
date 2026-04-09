#!/usr/bin/env python3
# 6h_elder_ray_regime_v1
# Hypothesis: 6h strategy using Elder Ray Index (Bull Power/Bear Power) combined with 1d ADX regime filter.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. Enter long when Bull Power > 0 and rising, ADX > 25 (trending).
# Enter short when Bear Power < 0 and falling, ADX > 25. Exit when power crosses zero or ADX < 20 (range).
# Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull/bear via ADX regime: trends captured, ranges avoided to prevent whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA(13)
    bear_power = ema13 - low   # EMA(13) - Low
    
    # Multi-timeframe: 1d ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ADX calculation (14-period)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray momentum (current vs previous)
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative or ADX drops into ranging
            if bull_power[i] <= 0 or ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive or ADX drops into ranging
            if bear_power[i] >= 0 or ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power positive AND rising AND trending
            if bull_power[i] > 0 and bull_rising and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power negative AND falling AND trending
            elif bear_power[i] < 0 and bear_falling and trending:
                position = -1
                signals[i] = -0.25
    
    return signals