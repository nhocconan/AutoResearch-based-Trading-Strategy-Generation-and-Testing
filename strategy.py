#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction and volume confirmation capture institutional momentum in both bull and bear markets. Weekly pivot provides long-term bias, Donchian breakout signals trend continuation, volume confirms conviction. Designed for low trade frequency (15-25/year) to minimize fee drag on 6h timeframe.
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
    
    # Get weekly data for pivot direction (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    weekly_pp = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = (2 * weekly_pp) - low_1w
    weekly_s1 = (2 * weekly_pp) - high_1w
    
    # Determine weekly bias: price above PP = bullish, below PP = bearish
    weekly_bullish = close_1w > weekly_pp
    weekly_bearish = close_1w < weekly_pp
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Get 12h data for volume context (optional filter)
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume']
    vol_ma_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Using pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection on 6h: volume > 1.8 * 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bullish_bias = weekly_bullish_aligned[i] > 0.5
        bearish_bias = weekly_bearish_aligned[i] > 0.5
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high with weekly bullish bias and volume spike
            if price > upper_channel and bullish_bias and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with weekly bearish bias and volume spike
            elif price < lower_channel and bearish_bias and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to Donchian low or weekly bias turns bearish
            if price < lower_channel or not bullish_bias:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to Donchian high or weekly bias turns bullish
            if price > upper_channel or not bearish_bias:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0