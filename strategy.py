#!/usr/bin/env python3
"""
6h Price Channel Breakout with Volume Spike and 12h ADX Trend Filter
Combines price channel breakouts (20-period high/low) with volume confirmation
and 12h ADX trend strength filter to avoid choppy markets.
Long when price breaks above 20-period high with volume spike and 12h ADX > 25.
Short when price breaks below 20-period low with volume spike and 12h ADX > 25.
Exits on opposite channel break or volume drop.
Designed for 6h timeframe to capture medium-term trends with controlled trade frequency.
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
    
    # Get 12h data for ADX trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
        # Subsequent values
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (values[i] - result[i-1])
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Price channel (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_strong = adx_12h_aligned[i] > 25  # Strong trend filter
        
        if position == 0:
            # Long: price breaks above channel high with volume spike and strong trend
            if (price > high_max[i] and volume_spike[i] and adx_strong):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below channel low with volume spike and strong trend
            elif (price < low_min[i] and volume_spike[i] and adx_strong):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below channel low OR volume drops AND trend weakens
            if (price < low_min[i]) or (not volume_spike[i] and adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above channel high OR volume drops AND trend weakens
            if (price > high_max[i]) or (not volume_spike[i] and adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PriceChannelBreakout_12hADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0