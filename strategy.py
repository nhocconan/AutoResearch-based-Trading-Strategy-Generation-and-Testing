#!/usr/bin/env python3
"""
12h_EhlersFisher_VolumeSpike_Trend_v1
Strategy: 12h Ehlers Fisher Transform (10-period) with volume spike confirmation and 1D EMA50/EMA200 trend filter.
Long: Fisher crosses above -1.5 in uptrend + volume spike. Short: Fisher crosses below +1.5 in downtrend + volume spike.
Exit on trend reversal or opposite signal. Uses Ehlers Fisher for early reversal detection in both bull/bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ehlers_fisher_transform(price, length):
    """Ehlers Fisher Transform"""
    price = np.asarray(price)
    n = len(price)
    if n < length:
        return np.full(n, np.nan)
    
    # Normalize price to [-1, 1] range
    highest = np.maximum.accumulate(price)
    lowest = np.minimum.accumulate(price)
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1, range_val)  # avoid division by zero
    
    # Avoid look-ahead: use only past data for normalization
    value = np.full(n, np.nan)
    for i in range(length-1, n):
        # Use only data up to i for highest/lowest calculation
        segment_high = np.max(price[i-length+1:i+1])
        segment_low = np.min(price[i-length+1:i+1])
        segment_range = segment_high - segment_low
        if segment_range == 0:
            normalized = 0
        else:
            normalized = 2 * (price[i] - segment_low) / segment_range - 1
        # Clamp to [-0.999, 0.999] to avoid infinity in log
        normalized = np.clip(normalized, -0.999, 0.999)
        # Fisher transform
        value[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    return value

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily data to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Ehlers Fisher Transform on 12h price (typical price)
    typical_price = (high + low + close) / 3.0
    fisher = ehlers_fisher_transform(typical_price, 10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(fisher[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation (current volume > 1.5x 20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > 1.5 * vol_ma
        else:
            vol_confirm = False
        
        # Fisher signals
        fisher_cross_up = (i > 0 and fisher[i-1] <= -1.5 and fisher[i] > -1.5)
        fisher_cross_down = (i > 0 and fisher[i-1] >= 1.5 and fisher[i] < 1.5)
        
        if position == 0:
            # Long: uptrend + volume + Fisher cross above -1.5
            if uptrend and vol_confirm and fisher_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + Fisher cross below +1.5
            elif downtrend and vol_confirm and fisher_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or Fisher cross below +1.5
            if not uptrend or fisher_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or Fisher cross above -1.5
            if not downtrend or fisher_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EhlersFisher_VolumeSpike_Trend_v1"
timeframe = "12h"
leverage = 1.0