#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (ATR ratio > 1.2). Donchian channels provide robust structure for breakouts in both bull and bear markets. 1d trend filter ensures alignment with higher timeframe momentum. Volume confirmation filters weak breakouts. Discrete sizing 0.25 limits trades to target 12-37/year. Works in bull/bear via 1d trend filter.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Donchian Channel (20-period) for breakouts
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and ATR ratio, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.2  # volume confirmation
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume confirmation AND aligned with 1d EMA50 trend
        # Long: price breaks above upper Donchian band (bullish breakout)
        # Short: price breaks below lower Donchian band (bearish breakout)
        long_entry = (close_val > upper_band) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < lower_band) and vol_spike and (close_val < ema_50_val)
        
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
            if close_val < upper_band and close_val > lower_band:  # back inside channel
                signals[i] = 0.0
                position = 0
            elif close_val < ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Donchian channel or trend reversal
            if close_val > lower_band and close_val < upper_band:  # back inside channel
                signals[i] = 0.0
                position = 0
            elif close_val > ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0