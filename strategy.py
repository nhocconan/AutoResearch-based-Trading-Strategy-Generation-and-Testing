#!/usr/bin/env python3
"""
12h_Wilson_Weekly_Bollinger_Upper_Lower_Band
Hypothesis: Trade breakouts above/below weekly Bollinger Bands in direction of daily EMA(34) trend, confirmed by volume >1.5x 12-period average. Uses daily trend filter to avoid counter-trend trades. Position size 0.25 targeting ~15 trades/year to minimize fee drift. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Daily EMA trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Weekly Bollinger Bands (20-period, 2 std dev)
    close_1w = df_1w['close'].values
    bb_period = 20
    bb_std = 2.0
    upper_band = np.full_like(close_1w, np.nan)
    lower_band = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= bb_period:
        for i in range(bb_period - 1, len(close_1w)):
            slice_data = close_1w[i - bb_period + 1:i + 1]
            sma = np.mean(slice_data)
            std = np.std(slice_data)
            upper_band[i] = sma + bb_std * std
            lower_band[i] = sma - bb_std * std
    
    # Align daily EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Align weekly Bollinger Bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation: volume > 1.5x 12-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 12
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume and above daily EMA
            if close[i] > upper_band_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and below daily EMA
            elif close[i] < lower_band_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower band (reverse signal) or below daily EMA
            if close[i] < lower_band_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper band (reverse signal) or above daily EMA
            if close[i] > upper_band_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Wilson_Weekly_Bollinger_Upper_Lower_Band"
timeframe = "12h"
leverage = 1.0