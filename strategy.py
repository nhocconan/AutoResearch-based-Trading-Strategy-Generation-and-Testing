#!/usr/bin/env python3
"""
12h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phases. 
Breakouts above Lips in uptrend (price>1d EMA50) or below Jaw in downtrend (price<1d EMA50) 
with volume confirmation capture strong moves. Designed for 12h timeframe to target 
50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
Works in both bull (trend following) and bear (mean reversion at extremes) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:  # Need 13 for Alligator (max period)
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    median_12h = (df_12h['high'] + df_12h['low']) / 2
    close_12h = df_12h['close']
    
    # Jaw (13-period SMMA, shifted by 8 bars)
    jaw = median_12h.rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth (8-period SMMA, shifted by 5 bars)
    teeth = median_12h.rolling(window=8, min_periods=8).mean().shift(5)
    # Lips (5-period SMMA, shifted by 3 bars)
    lips = median_12h.rolling(window=5, min_periods=5).mean().shift(3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw.values, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth.values, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips.values, additional_delay_bars=0)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Alligator, and volume MA
    start_idx = max(51, 13 + 8, 20)  # 51 for EMA50, 21 for Jaw (13+8), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        jaw_val = jaw_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for long breakouts above Lips
                long_signal = (curr_close > lips_val) and volume_confirm
            else:
                # Downtrend: look for short breakdowns below Jaw
                short_signal = (curr_close < jaw_val) and volume_confirm
            
            # In ranging markets (price between Jaw/Lips), fade extremes
            if not price_above_ema and not price_below_ema:
                # Actually, this case is covered by above/below - price exactly at EMA is rare
                # Add explicit ranging condition: price between Jaw and Lips
                in_range = (curr_close >= jaw_val) and (curr_close <= lips_val)
                if in_range:
                    # Fade extremes: long near Jaw, short near Lips
                    long_signal = (curr_close <= jaw_val * 1.002) and volume_confirm  # near Jaw
                    short_signal = (curr_close >= lips_val * 0.998) and volume_confirm  # near Lips
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            # Clear signal flags for next iteration
            if 'long_signal' in locals():
                del long_signal
            if 'short_signal' in locals():
                del short_signal
        elif position == 1:
            # Exit long: price breaks below Jaw or reverses below EMA
            if curr_close < jaw_val or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Lips or reverses above EMA
            if curr_close > lips_val or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0