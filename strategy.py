#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
    # Donchian breakouts capture institutional participation; volume confirms validity.
    # Chop regime (CHOP > 61.8) avoids whipsaws in ranging markets; breakouts only in trending regimes.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d chop regime (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = np.zeros(len(close_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    for i in range(1, len(close_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr14) / (max_high - min_low)) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop = np.where(
        (range_14 > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (avoid breakouts in chop)
        # We want chop <= 61.8 for trending market where breakouts work
        regime_filter = chop_aligned[i] <= 61.8
        
        # Donchian breakout conditions
        long_breakout = (close[i] > donchian_h[i-1]) and volume_filter and regime_filter
        short_breakout = (close[i] < donchian_l[i-1]) and volume_filter and regime_filter
        
        # Exit conditions: return to midpoint of Donchian channel
        midpoint = (donchian_h[i] + donchian_l[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0