#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: Trade Camarilla H3/L3 breakouts on 4h with 1d EMA50 trend filter and volume confirmation.
Only trade when 1d trend is aligned (price > EMA50 for long, price < EMA50 for short) AND volume > 1.5x 20-period average.
Exit on opposite Camarilla level touch or trend reversal. Uses H3/L3 levels for stronger breakouts.
Position size: 0.25. Target: 25-40 trades/year to stay under 400-trade 4h hard max.
Works in bull (breakouts with trend) and bear (strong breakdowns with trend) markets.
Volume confirmation reduces false breakouts in low-participation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_h3_1d = c_1d + (range_1d * 1.1 / 4.0)   # H3 level
    camarilla_l3_1d = c_1d - (range_1d * 1.1 / 4.0)   # L3 level
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + 1d uptrend + volume confirmation
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla L3 + 1d downtrend + volume confirmation
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1d_bearish and volume_confirm
            
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
            # Exit: price touches Camarilla L3 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0