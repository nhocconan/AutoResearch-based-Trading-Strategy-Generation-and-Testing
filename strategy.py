#!/usr/bin/env python3
"""
6h_WilliamsAlligator_RegimeBreakout
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) defines trend regime on 1d HTF; trade 6h breakouts of Donchian(20) only when price is outside Alligator mouth (trending) and aligned with Alligator direction. Uses volume confirmation to avoid false breakouts. Designed for low-frequency, high-conviction entries in both bull and bear markets by filtering chop via Alligator convergence.
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
    
    # Load 1d data ONCE before loop for HTF Alligator and Donchian context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs of median price (typical price)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    jaw = pd.Series(typical_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # Jaw: 13-period, 8-shift
    teeth = pd.Series(typical_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values   # Teeth: 8-period, 5-shift
    lips = pd.Series(typical_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values    # Lips: 5-period, 3-shift
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Donchian(20) on 6h for breakout signals
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = np.concatenate([np.full(19, np.nan), high_roll[19:]])  # align to current bar
    donchian_low = np.concatenate([np.full(19, np.nan), low_roll[19:]])
    
    # Volume confirmation: volume > 60th percentile of 30-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_60 = vol_series.rolling(window=30, min_periods=30).quantile(0.60).values
    volume_confirm = volume > vol_percentile_60
    
    # Fixed position size for low turnover
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for 1d indicators, 20 for Donchian, 30 for volume)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_percentile_60[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        # Alligator regime: trending when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        # Avoid trading when Alligator lines are intertwined (choppy/market asleep)
        bullish_regime = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_regime = (lips_val < teeth_val) and (teeth_val < jaw_val)
        trending_regime = bullish_regime or bearish_regime
        
        # Entry conditions: Donchian breakout with volume confirmation AND aligned with Alligator trend
        long_entry = (close_val > donchian_high_val) and vol_conf and bullish_regime
        short_entry = (close_val < donchian_low_val) and vol_conf and bearish_regime
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = fixed_size
                position = 1
            elif short_entry:
                signals[i] = -fixed_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price retouches lips (Alligator wake-up signal) or close below teeth
            if close_val < lips_val or close_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit when price retouches lips or close above teeth
            if close_val > lips_val or close_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "6h_WilliamsAlligator_RegimeBreakout"
timeframe = "6h"
leverage = 1.0