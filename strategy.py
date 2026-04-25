#!/usr/bin/env python3
"""
6h Camarilla H4/L4 Breakout with Weekly EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla H4/L4 levels represent stronger breakout points than H3/L3. Combined with weekly EMA50 trend filter (bull/bear regime from higher timeframe) and volume spike (>1.8x 20-bar vol MA) to capture strong momentum moves. Weekly EMA50 provides stable trend definition less prone to whipsaw. Targeting 15-25 trades per year on 6h to minimize fee drag while maintaining edge in both bull and bear markets through regime-adaptive breakouts.
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
    
    # Get weekly data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 prior day
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly EMA50, Camarilla, and volume MA
    start_idx = max(51, 20)  # 51 for weekly EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        h4_val = camarilla_h4_aligned[i]
        l4_val = camarilla_l4_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above H4 + price above weekly EMA50 + volume confirmation
            long_signal = (curr_high > h4_val) and price_above_ema and volume_confirm
            # Short: break below L4 + price below weekly EMA50 + volume confirmation
            short_signal = (curr_low < l4_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below H4 OR price crosses below weekly EMA50
            if (curr_close < h4_val) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above L4 OR price crosses above weekly EMA50
            if (curr_close > l4_val) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0