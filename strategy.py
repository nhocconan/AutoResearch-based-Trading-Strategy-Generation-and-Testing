#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray and volume confirmation
# Uses 1-day Elder Ray (bull/bear power) for trend direction, 6-hour Williams Alligator
# (JAW/TEETH/LIPS) for entry signals, and volume confirmation to filter weak signals.
# Works in bull markets by buying on bullish Alligator alignment with positive bull power,
# and in bear markets by selling on bearish Alligator alignment with negative bear power.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while
# capturing trending moves with proper risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray (bull/bear power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_len = 13
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_1d
    bear_power = low_1d - ema_1d
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    # Using EMA as proxy for SMMA for computational efficiency (similar smoothing effect)
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Jaw: 13-period EMA of median price, shifted 8 bars
    median_price_6h = (high_6h + low_6h) / 2
    jaw_len = 13
    jaw_shift = 8
    jaw = np.full(len(close_6h), np.nan)
    if len(median_price_6h) >= jaw_len:
        multiplier = 2 / (jaw_len + 1)
        jaw[jaw_len-1] = np.mean(median_price_6h[:jaw_len])
        for i in range(jaw_len, len(median_price_6h)):
            jaw[i] = (median_price_6h[i] * multiplier) + (jaw[i-1] * (1 - multiplier))
        # Apply shift
        jaw = np.roll(jaw, jaw_shift)
        jaw[:jaw_shift] = np.nan
    
    # Teeth: 8-period EMA of median price, shifted 5 bars
    teeth_len = 8
    teeth_shift = 5
    teeth = np.full(len(close_6h), np.nan)
    if len(median_price_6h) >= teeth_len:
        multiplier = 2 / (teeth_len + 1)
        teeth[teeth_len-1] = np.mean(median_price_6h[:teeth_len])
        for i in range(teeth_len, len(median_price_6h)):
            teeth[i] = (median_price_6h[i] * multiplier) + (teeth[i-1] * (1 - multiplier))
        # Apply shift
        teeth = np.roll(teeth, teeth_shift)
        teeth[:teeth_shift] = np.nan
    
    # Lips: 5-period EMA of median price, shifted 3 bars
    lips_len = 5
    lips_shift = 3
    lips = np.full(len(close_6h), np.nan)
    if len(median_price_6h) >= lips_len:
        multiplier = 2 / (lips_len + 1)
        lips[lips_len-1] = np.mean(median_price_6h[:lips_len])
        for i in range(lips_len, len(median_price_6h)):
            lips[i] = (median_price_6h[i] * multiplier) + (lips[i-1] * (1 - multiplier))
        # Apply shift
        lips = np.roll(lips, lips_shift)
        lips[:lips_shift] = np.nan
    
    # Calculate 20-period average volume on 6h for spike detection
    vol_6h = df_6h['volume'].values
    vol_ma_6h = np.full(len(vol_6h), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_6h)):
        vol_ma_6h[i] = np.mean(vol_6h[i-vol_period:i])
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period: max of all lookbacks plus shifts
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h_aligned[i] if vol_ma_6h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Williams Alligator signals:
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + positive bull power + volume
            if bullish_alignment and bull_power_aligned[i] > 0 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Bearish Alligator alignment + negative bear power + volume
            elif bearish_alignment and bear_power_aligned[i] < 0 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR bull power turns negative
            if bearish_alignment or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR bear power turns positive
            if bullish_alignment or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_1dElderRay_Volume"
timeframe = "6h"
leverage = 1.0