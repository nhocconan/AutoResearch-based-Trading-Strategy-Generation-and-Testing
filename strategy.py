#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeRegime
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume regime filter (ATR ratio > 1.3). Donchian channels provide clear breakout levels. 12h EMA50 ensures alignment with higher timeframe momentum to avoid counter-trend trades. Volume regime confirms institutional participation. Discrete sizing 0.25 limits trades (~15-30/year). Works in bull/bear via 12h trend filter - only trade in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Donchian Channel (20 periods)
    period_dc = 20
    dc_upper = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    dc_lower = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for ATR ratio and EMA, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        dc_upper_val = dc_upper[i]
        dc_lower_val = dc_lower[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = atr_ratio[i] > 1.3  # volume regime
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume spike AND aligned with 12h EMA50 trend
        # Long: price breaks above upper Donchian band
        # Short: price breaks below lower Donchian band
        long_entry = (close_val > dc_upper_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < dc_lower_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Donchian channel or trend reversal
            if close_val < dc_upper_val and close_val > dc_lower_val:  # back inside channel
                signals[i] = 0.0
                position = 0
            elif close_val < ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Donchian channel or trend reversal
            if close_val > dc_lower_val and close_val < dc_upper_val:  # back inside channel
                signals[i] = 0.0
                position = 0
            elif close_val > ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0