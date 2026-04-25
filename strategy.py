#!/usr/bin/env python3
"""
1d_Alligator_WeeklyTrend_VolumeFilter_v1
Hypothesis: Use Williams Alligator on daily timeframe with 1-week trend filter and volume confirmation.
Alligator (Smoothed Medians) identifies trend direction and strength; 1-week EMA filters for higher-timeframe alignment;
volume spike confirms institutional interest. Discrete sizing (0.25) and minimum holding period (3 days) reduce fee drag.
Designed to work in both bull (trend following) and bear (mean-reversion at extremes) markets via regime-aware exits.
Target: 15-25 trades/year to minimize fee drag while capturing medium-term swings.
"""

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
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams Alligator: three smoothed medians (Jaw, Teeth, Lips)
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe (already aligned via get_htf_data, but ensure no look-ahead)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR for volume spike filter and stoploss
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for Alligator (lips: 5+3=8), 1w EMA20 (20), ATR (14)
    start_idx = max(8, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 2.0 * ATR (adaptive threshold for institutional interest)
        volume_spike = volume[i] > 2.0 * atr[i]
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            
            # Long: bullish alignment AND price above lips AND 1w trend up (close > EMA20) AND volume spike
            long_setup = bullish_alignment and (close[i] > lips_aligned[i]) and \
                         (close[i] > ema_20_1w_aligned[i]) and volume_spike
            # Short: bearish alignment AND price below lips AND 1w trend down (close < EMA20) AND volume spike
            short_setup = bearish_alignment and (close[i] < lips_aligned[i]) and \
                          (close[i] < ema_20_1w_aligned[i]) and volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: Alligator reverses (lips < teeth) OR 1w trend turns down OR min holding period (3 days)
            if (lips_aligned[i] < teeth_aligned[i]) or \
               (close[i] < ema_20_1w_aligned[i]) or \
               (bars_since_entry >= 3):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: Alligator reverses (lips > teeth) OR 1w trend turns up OR min holding period (3 days)
            if (lips_aligned[i] > teeth_aligned[i]) or \
               (close[i] > ema_20_1w_aligned[i]) or \
               (bars_since_entry >= 3):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "1d_Alligator_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0