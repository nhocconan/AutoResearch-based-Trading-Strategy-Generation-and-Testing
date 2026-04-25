#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade 12h Camarilla H3/L3 breakouts with 1w EMA50 trend filter and volume confirmation.
Designed for 12-30 trades/year on 12h timeframe to minimize fee drag while capturing strong directional moves.
Uses higher timeframe trend (1w) and wider Camarilla levels (H3/L3) for fewer, higher-quality signals.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla H3 and L3 levels (wider breakout levels)
    camarilla_h3 = typical_price_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = typical_price_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA50 (50) and ensure Camarilla data is ready
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.5 * average of last 28 periods (28*12h = 14 days)
        if i >= 28:
            vol_avg = np.mean(volume[i-28:i])
            volume_spike = volume[i] > 2.5 * vol_avg
        else:
            volume_spike = False
        
        if position == 0:
            # Long: price breaks above H3 AND 1w trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike
            # Short: price breaks below L3 AND 1w trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters between L3 and H3 OR 1w trend turns bearish
            if (camarilla_l3_aligned[i] < close[i] < camarilla_h3_aligned[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between L3 and H3 OR 1w trend turns bullish
            if (camarilla_l3_aligned[i] < close[i] < camarilla_h3_aligned[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0