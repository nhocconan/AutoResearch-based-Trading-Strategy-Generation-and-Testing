#!/usr/bin/env python3
"""
12h Williams Alligator + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence on 12h.
In trending markets (price outside Alligator's mouth), Alligator lines converge/diverge with momentum.
Combined with 1w EMA50 trend filter and volume spike confirmation, captures strong trending moves.
Works in both bull/bear markets by trend-filtering Alligator breakouts.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high + low) / 2.0
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculation (13) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator conditions:
        # Trending market: Lips > Teeth > Jaw (uptrend) OR Lips < Teeth < Jaw (downtrend)
        # Mouth open: divergence between Lips and Jaw
        lips_above_jaw = lips_val > jaw_val
        lips_below_jaw = lips_val < jaw_val
        teeth_between = (lips_val > teeth_val > jaw_val) or (lips_val < teeth_val < jaw_val)
        
        # Strong trend: price outside Alligator's mouth + aligned with 1w EMA50
        strong_uptrend = lips_above_jaw and teeth_between and (curr_close > ema_trend)
        strong_downtrend = lips_below_jaw and teeth_between and (curr_close < ema_trend)
        
        # Entry signals with volume confirmation
        if position == 0:
            if strong_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            elif strong_downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakening (Lips crosses below Teeth) or price breaks below Jaw
            if lips_val < teeth_val or curr_close < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening (Lips crosses above Teeth) or price breaks above Jaw
            if lips_val > teeth_val or curr_close > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0