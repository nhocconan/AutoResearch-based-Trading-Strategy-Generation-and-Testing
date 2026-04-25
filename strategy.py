#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Alligator (SMAs with offset) identifies trend alignment, 
1d EMA34 filters higher timeframe trend, volume spike confirms institutional participation, 
and chop filter avoids ranging markets. Works in bull/bear via trend filter.
Target: 12-37 trades/year on 6h.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # All lines are SMAs with forward shift
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter (avoid ranging markets)
    # Simplified: ATR(14) * 14 / (max(high,14) - min(low,14))
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    price_range_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / price_range_14) / np.log10(14)
    chop_filter = chop > 50  # Only trade when chop > 50 (not too trending, not too choppy)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators (max shift is 8 for jaw)
    start_idx = max(13, 8, 5, 20, 14) + 8  # +8 for jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + 1d trend + volume + chop filter
            long_entry = uptrend and (curr_close > ema_34_1d_aligned[i]) and vol_spike and chop_ok
            short_entry = downtrend and (curr_close < ema_34_1d_aligned[i]) and vol_spike and chop_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator reverses (Lips < Jaw) OR 1d trend reverses
            if lips[i] < jaw[i] or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator reverses (Lips > Jaw) OR 1d trend reverses
            if lips[i] > jaw[i] or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0